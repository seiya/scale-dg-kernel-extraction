// Lower-bound cost of an Ozaki-Scheme-II style INT8 emulation for the p=255
// volume GEMMs of this repository.
//
// Shapes are taken from cuda_cal_dqdt_gemm in mod_cuda_dg_kernels.cuf with
// Nq = 256, Ne = 1:
//   x-deriv : C(Nq, nq2*Ne)      = D1D(Nq,Nq)      * flux_x(Nq, nq2*Ne)
//   y-deriv : batched, batch = Nq*Ne planes, each (Nq,Nq) * (Nq,Nq)
//
// Reported:
//   [native]   cublasDgemm at the same shape (the thing to beat)
//   [int8 x s] s INT8 GEMMs, all accumulating into one INT32 C
//              -> pure arithmetic floor, no emulation traffic at all
//   [int8 x s, s buffers] s INT8 GEMMs into s distinct INT32 C buffers
//              -> adds the O(s*M*N) partial-product store traffic
//   [reconstruct] one pass reading the s INT32 buffers, writing the FP64 C
//              -> prices the CRT reconstruction traffic
//
// Decision rule: if [int8 x s] alone already exceeds [native], the scheme
// cannot pay off at this K, whatever the reconstruction costs.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  exit(1); } } while (0)

#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
  exit(1); } } while (0)

static const int NQ  = 256;
static const int NQ2 = NQ * NQ;

// One CRT-style reconstruction pass: read s INT32 residues per output point,
// combine them with per-modulus FP64 weights, write one FP64 value.  The
// arithmetic is a placeholder; the point is the memory traffic, which is what
// any Ozaki-II reconstruction has to move.
__global__ void reconstruct_kernel(double *__restrict__ out,
                                   const int *const *__restrict__ residues,
                                   const double *__restrict__ weight,
                                   int s, long long n)
{
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
       i < n; i += (long long)gridDim.x * blockDim.x) {
    double acc = 0.0;
    for (int j = 0; j < s; ++j) acc += weight[j] * (double)residues[j][i];
    out[i] = acc;
  }
}

__global__ void fill_i8(signed char *p, long long n, int seed)
{
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
       i < n; i += (long long)gridDim.x * blockDim.x)
    p[i] = (signed char)((i * 1103515245LL + seed) % 127);
}

__global__ void fill_f64(double *p, long long n, int seed)
{
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
       i < n; i += (long long)gridDim.x * blockDim.x)
    p[i] = (double)((i + seed) % 17) * 0.125;
}

struct Timer {
  cudaEvent_t a, b;
  Timer()  { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
  ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
  void start() { CUDA_CHECK(cudaEventRecord(a)); }
  float stop() { float ms; CUDA_CHECK(cudaEventRecord(b));
                 CUDA_CHECK(cudaEventSynchronize(b));
                 CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); return ms; }
};

// Median of `iters` timed repetitions of `body`, in microseconds.
template <class F>
static double bench(F body, int iters, int warmup)
{
  Timer t;
  for (int i = 0; i < warmup; ++i) body();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> us;
  for (int i = 0; i < iters; ++i) {
    t.start();
    body();
    us.push_back(t.stop() * 1000.0);
  }
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

int main(int argc, char **argv)
{
  const int s  = (argc > 1) ? atoi(argv[1]) : 14;   // number of moduli
  const int Ne = (argc > 2) ? atoi(argv[2]) : 1;

  const int M = NQ;              // x GEMM
  const long long N = (long long)NQ2 * Ne;
  const int K = NQ;
  const long long MN = (long long)M * N;

  const int    plane_batch = NQ * Ne;               // y GEMM
  const long long y_flops  = 2.0 * NQ * NQ * NQ * (long long)plane_batch;

  printf("# Ozaki-II INT8 floor test\n");
  printf("# x GEMM  M=%d N=%lld K=%d   (C = %.1f MB fp64 / %.1f MB int32)\n",
         M, N, K, MN * 8 / 1.0e6, MN * 4 / 1.0e6);
  printf("# y GEMM  %dx%dx%d batched x %d\n", NQ, NQ, NQ, plane_batch);
  printf("# s = %d moduli, Ne = %d\n\n", s, Ne);

  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));

  // ---- buffers -------------------------------------------------------------
  double *dA, *dB, *dC;                       // fp64 reference
  CUDA_CHECK(cudaMalloc(&dA, (size_t)K * M * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)MN    * sizeof(double)));
  fill_f64<<<256, 256>>>(dA, (long long)K * M, 1);
  fill_f64<<<1024, 256>>>(dB, (long long)K * N, 2);

  signed char *iA, *iB;                       // int8 operands
  CUDA_CHECK(cudaMalloc(&iA, (size_t)K * M));
  CUDA_CHECK(cudaMalloc(&iB, (size_t)K * N));
  fill_i8<<<256, 256>>>(iA, (long long)K * M, 3);
  fill_i8<<<1024, 256>>>(iB, (long long)K * N, 5);

  int *iC1;                                   // single int32 accumulator
  CUDA_CHECK(cudaMalloc(&iC1, (size_t)MN * sizeof(int)));

  std::vector<int *> hC(s);                   // s distinct int32 accumulators
  for (int j = 0; j < s; ++j)
    CUDA_CHECK(cudaMalloc(&hC[j], (size_t)MN * sizeof(int)));
  int **dCs;
  CUDA_CHECK(cudaMalloc(&dCs, s * sizeof(int *)));
  CUDA_CHECK(cudaMemcpy(dCs, hC.data(), s * sizeof(int *), cudaMemcpyHostToDevice));

  double *dW;
  CUDA_CHECK(cudaMalloc(&dW, s * sizeof(double)));
  fill_f64<<<1, 64>>>(dW, s, 7);
  CUDA_CHECK(cudaDeviceSynchronize());

  const double d_one = 1.0, d_zero = 0.0;
  const int    i_one = 1,   i_zero = 0;

  const double x_flops = 2.0 * M * (double)N * K;

  // ---- 1. native DGEMM, x shape -------------------------------------------
  double t_dx = bench([&] {
    CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, (int)N, K,
                             &d_one, dA, M, dB, K, &d_zero, dC, M));
  }, 20, 5);
  printf("native  DGEMM   x        : %8.1f us   %6.2f TFLOP/s\n",
         t_dx, x_flops / (t_dx * 1.0e-6) / 1.0e12);

  // ---- 2. s INT8 GEMMs into one C (arithmetic floor) -----------------------
  // INT8 on cuBLAS requires the TN layout (op(A)=T, op(B)=N).
  double t_i1 = bench([&] {
    for (int j = 0; j < s; ++j)
      CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, (int)N, K,
                                &i_one, iA, CUDA_R_8I, K, iB, CUDA_R_8I, K,
                                &i_zero, iC1, CUDA_R_32I, M,
                                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
  }, 20, 5);
  printf("int8 x %-2d GEMM  x, 1 C   : %8.1f us   (%6.1f us each, %7.1f TOP/s)\n",
         s, t_i1, t_i1 / s, s * x_flops / (t_i1 * 1.0e-6) / 1.0e12);

  // ---- 3. s INT8 GEMMs into s distinct C ----------------------------------
  double t_is = bench([&] {
    for (int j = 0; j < s; ++j)
      CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, (int)N, K,
                                &i_one, iA, CUDA_R_8I, K, iB, CUDA_R_8I, K,
                                &i_zero, hC[j], CUDA_R_32I, M,
                                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
  }, 20, 5);
  printf("int8 x %-2d GEMM  x, %2d C  : %8.1f us\n", s, s, t_is);

  // ---- 4. CRT reconstruction pass -----------------------------------------
  double t_rc = bench([&] {
    reconstruct_kernel<<<2048, 256>>>(dC, dCs, dW, s, MN);
  }, 20, 5);
  {
    double bytes = (double)MN * (4.0 * s + 8.0);
    printf("reconstruct     x        : %8.1f us   %6.2f TB/s (%.0f MB)\n",
           t_rc, bytes / (t_rc * 1.0e-6) / 1.0e12, bytes / 1.0e6);
  }

  printf("\nx total emulated (%d C + reconstruct): %8.1f us  vs native %8.1f us  -> %.2fx\n",
         s, t_is + t_rc, t_dx, (t_is + t_rc) / t_dx);
  printf("x total fused-ideal (1 C + reconstruct): %8.1f us  vs native %8.1f us  -> %.2fx\n\n",
         t_i1 + t_rc, t_dx, (t_i1 + t_rc) / t_dx);

  // ---- 5. batched y shape --------------------------------------------------
  const long long sp = (long long)NQ * NQ;
  double t_dy = bench([&] {
    CUBLAS_CHECK(cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                           NQ, NQ, NQ, &d_one,
                                           dB, NQ, sp, dA, NQ, 0,
                                           &d_zero, dC, NQ, sp, plane_batch));
  }, 20, 5);
  printf("native  DGEMM   y batched: %8.1f us   %6.2f TFLOP/s\n",
         t_dy, y_flops / (t_dy * 1.0e-6) / 1.0e12);

  double t_iy = bench([&] {
    for (int j = 0; j < s; ++j)
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
                                              NQ, NQ, NQ, &i_one,
                                              iB, CUDA_R_8I, NQ, sp,
                                              iA, CUDA_R_8I, NQ, 0,
                                              &i_zero, iC1, CUDA_R_32I, NQ, sp,
                                              plane_batch,
                                              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
  }, 20, 5);
  printf("int8 x %-2d GEMM  y batched: %8.1f us   (%6.1f us each)\n", s, t_iy, t_iy / s);
  printf("\ny int8-only vs native: %.2fx\n", t_iy / t_dy);

  cublasDestroy(h);
  return 0;
}
