// At which polynomial order could an Ozaki-II INT8 emulation beat native
// FP64 for the p=255-style volume GEMM of this repository?
//
// The x GEMM has M = Nq, N = Nq^2 * Ne, K = Nq with Nq = p+1.  For the
// emulation the decisive ratio is
//     useful ops / INT32 output bytes = 2*M*N*K / (4*M*N) = K/2 = Nq/2,
// which does not depend on N.  So the achieved rates R_fp64(Nq) and
// R_int8(Nq) can be measured at a capped N and extrapolated to N = Nq^2.
//
// Model, per tendency call, for the x GEMM:
//     t_native = 2*Nq^4 / R_fp64(Nq)
//     t_int8   = s * 2*Nq^4 / R_int8(Nq)
//     t_recon  = (4*s + 8) * Nq^3 / BW
// Break-even s is where t_int8 + t_recon = t_native.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("cuda %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
#define BK(x) do { cublasStatus_t st=(x); if(st!=CUBLAS_STATUS_SUCCESS){printf("cublas %d @%d\n",(int)st,__LINE__);exit(1);} } while(0)

static const int NCAP = 65536;   // capped N used for rate measurement

template <class F> static double med(F body, int iters, int warm)
{
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i = 0; i < warm; ++i) body();
  CK(cudaDeviceSynchronize());
  std::vector<double> us;
  for (int i = 0; i < iters; ++i) {
    float ms; CK(cudaEventRecord(a)); body();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b)); us.push_back(ms * 1000.0);
  }
  std::sort(us.begin(), us.end());
  cudaEventDestroy(a); cudaEventDestroy(b);
  return us[us.size() / 2];
}

__global__ void reconstruct_kernel(double *__restrict__ out,
                                   const int *const *__restrict__ res,
                                   int s, long long n)
{
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
       i < n; i += (long long)gridDim.x * blockDim.x) {
    double acc = 0.0;
    for (int j = 0; j < s; ++j) acc += 1.5 * (double)res[j][i];
    out[i] = acc;
  }
}

int main(int argc, char **argv)
{
  const int s_ref = (argc > 1) ? atoi(argv[1]) : 14;
  cublasHandle_t h; BK(cublasCreate(&h));

  const int NQMAX = 16384;
  double *dA, *dB, *dC; signed char *iA, *iB; int *iC;
  CK(cudaMalloc(&dA, (size_t)NQMAX * NQMAX * sizeof(double)));
  CK(cudaMalloc(&dB, (size_t)NQMAX * NCAP  * sizeof(double)));
  CK(cudaMalloc(&dC, (size_t)NQMAX * NCAP  * sizeof(double)));
  CK(cudaMalloc(&iA, (size_t)NQMAX * NQMAX));
  CK(cudaMalloc(&iB, (size_t)NQMAX * NCAP));
  CK(cudaMalloc(&iC, (size_t)NQMAX * NCAP  * sizeof(int)));
  CK(cudaMemset(dA, 0x3f, (size_t)NQMAX * NQMAX * sizeof(double)));
  CK(cudaMemset(dB, 0x3f, (size_t)NQMAX * NCAP  * sizeof(double)));
  CK(cudaMemset(iA, 1, (size_t)NQMAX * NQMAX));
  CK(cudaMemset(iB, 1, (size_t)NQMAX * NCAP));

  // ---- reconstruction bandwidth, measured once ----------------------------
  double BW;
  {
    const int s = s_ref;
    const long long n = 16777216;                    // 64 Mi points
    std::vector<int *> hp(s);
    for (int j = 0; j < s; ++j) CK(cudaMalloc(&hp[j], n * sizeof(int)));
    int **dp; CK(cudaMalloc(&dp, s * sizeof(int *)));
    CK(cudaMemcpy(dp, hp.data(), s * sizeof(int *), cudaMemcpyHostToDevice));
    double *out; CK(cudaMalloc(&out, n * sizeof(double)));
    double us = med([&]{ reconstruct_kernel<<<4096,256>>>(out, dp, s, n); }, 15, 5);
    double bytes = (double)n * (4.0 * s + 8.0);
    BW = bytes / (us * 1e-6);
    printf("# reconstruction bandwidth (s=%d, %.0f MB): %.2f TB/s\n\n", s, bytes/1e6, BW/1e12);
    for (int j = 0; j < s; ++j) cudaFree(hp[j]);
    cudaFree(dp); cudaFree(out);
  }

  const double d1 = 1.0, d0 = 0.0; const int i1 = 1, i0 = 0;

  printf("# rates measured at M=Nq, N=%d, K=Nq; extrapolated to N=Nq^2\n", NCAP);
  printf("#   p      Nq   R_fp64      R_int8    int8/fp64   t_nat[ms]   s=%d[ms]  ratio  break-even s\n", s_ref);
  printf("#             [TFLOP/s]    [TOP/s]      ratio      (N=Nq^2)   (N=Nq^2)\n");

  for (int Nq : {128, 256, 512, 576, 640, 704, 768, 896, 1024, 2048, 4096, 8192, 16384}) {
    double t_d = med([&]{
      BK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, Nq, NCAP, Nq,
                     &d1, dA, Nq, dB, Nq, &d0, dC, Nq));
    }, 11, 4);
    double t_i = med([&]{
      BK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, Nq, NCAP, Nq,
                      &i1, iA, CUDA_R_8I, Nq, iB, CUDA_R_8I, Nq,
                      &i0, iC, CUDA_R_32I, Nq, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    }, 11, 4);

    const double ops = 2.0 * Nq * (double)NCAP * Nq;
    const double Rd = ops / (t_d * 1e-6);
    const double Ri = ops / (t_i * 1e-6);

    // extrapolate to the real shape N = Nq^2 (Ne = 1)
    const double work = 2.0 * (double)Nq * Nq * Nq * Nq;   // 2*Nq^4
    const double t_nat = work / Rd;
    const double t_rc  = (4.0 * s_ref + 8.0) * (double)Nq * Nq * Nq / BW;
    const double t_emu = s_ref * work / Ri + t_rc;

    // break-even s:  s*work/Ri + (4s+8)*Nq^3/BW = work/Rd
    const double cube = (double)Nq * Nq * Nq;
    const double s_be = (work / Rd - 8.0 * cube / BW) / (work / Ri + 4.0 * cube / BW);

    printf("%6d %7d %9.2f %11.1f %10.1f %11.3f %10.3f %6.2f %10.1f\n",
           Nq - 1, Nq, Rd / 1e12, Ri / 1e12, Ri / Rd,
           t_nat * 1e3, t_emu * 1e3, t_emu / t_nat, s_be);
  }
  return 0;
}
