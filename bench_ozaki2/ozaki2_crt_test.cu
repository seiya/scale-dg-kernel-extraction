// Validate Ozaki-II CRT + pack against native DGEMM at p=255 volume shapes.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../cuda_ozaki2_gemm.h"

extern "C" int cublas_gemm_init(int emulate, int mantissa_fixed, int mantissa_bits);
cudaStream_t dg_cuda_stream = nullptr;

#define CK(...)                                                                  \
  do {                                                                         \
    cudaError_t e = (__VA_ARGS__);                                               \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "cuda %s @%d\n", cudaGetErrorString(e), __LINE__);     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define BK(...)                                                                  \
  do {                                                                         \
    cublasStatus_t st_ = (__VA_ARGS__);                                        \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cublas %d @%d\n", (int)st_, __LINE__);                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const int NQ = 256;
static const int NQ2 = NQ * NQ;

template <class F>
static double med(F body, int it, int warm)
{
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  for (int i = 0; i < warm; ++i) body();
  CK(cudaDeviceSynchronize());
  std::vector<double> us;
  for (int i = 0; i < it; ++i) {
    float ms;
    CK(cudaEventRecord(a));
    body();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    us.push_back(ms * 1000.0);
  }
  std::sort(us.begin(), us.end());
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return us[us.size() / 2];
}

static double max_abs_diff(const double *a, const double *b, long long n)
{
  double mx = 0.0;
  for (long long i = 0; i < n; ++i) {
    mx = fmax(mx, fabs(a[i] - b[i]));
  }
  return mx;
}

int main(int argc, char **argv)
{
  const int s = (argc > 1) ? atoi(argv[1]) : 14;
  const int Ne = (argc > 2) ? atoi(argv[2]) : 1;

  cublasHandle_t h;
  BK(cublasCreate(&h));

  if (ozaki2_init(s, 1) != 0) {
    fprintf(stderr, "ozaki2_init failed\n");
    return 1;
  }
  if (cublas_gemm_init(0, 1, 55) != 0) {
    fprintf(stderr, "cublas_gemm_init failed\n");
    return 1;
  }
  if (ozaki2_alloc_workspace(NQ, Ne, NQ * NQ * NQ) != 0) {
    fprintf(stderr, "ozaki2_alloc_workspace failed\n");
    return 1;
  }

  const long long MN = static_cast<long long>(NQ) * NQ2 * Ne;
  const int Ncols = NQ2 * Ne;
  const long long sp = static_cast<long long>(NQ) * NQ;
  const int plane_batch = NQ * Ne;

  double *dA, *dB, *dRef, *dOz;
  CK(cudaMalloc(&dA, static_cast<size_t>(NQ * NQ) * sizeof(double)));
  CK(cudaMalloc(&dB, static_cast<size_t>(MN) * sizeof(double)));
  CK(cudaMalloc(&dRef, static_cast<size_t>(MN) * sizeof(double)));
  CK(cudaMalloc(&dOz, static_cast<size_t>(MN) * sizeof(double)));

  std::vector<double> hA(NQ * NQ);
  std::vector<double> hB(MN);
  for (int i = 0; i < NQ * NQ; ++i) {
    hA[i] = sin(0.13 * i) * 0.5;
  }
  for (long long i = 0; i < MN; ++i) {
    hB[i] = cos(0.07 * i) * 0.25;
  }
  CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), static_cast<size_t>(MN) * sizeof(double),
                cudaMemcpyHostToDevice));

  const double d1 = 1.0, d0 = 0.0;
  printf("# ozaki2_crt_test s=%d Ne=%d\n", s, Ne);

  // 4x4 smoke test
  {
    const int T = 4;
    const double tol4 = 1e-2;
    double hAt[16], hBt[16];
    for (int i = 0; i < 16; ++i) {
      hAt[i] = 0.1 * (i + 1);
      hBt[i] = 0.05 * (i + 2);
    }
    double *dAt, *dBt, *dRt, *dOt;
    CK(cudaMalloc(&dAt, 16 * sizeof(double)));
    CK(cudaMalloc(&dBt, 16 * sizeof(double)));
    CK(cudaMalloc(&dRt, 16 * sizeof(double)));
    CK(cudaMalloc(&dOt, 16 * sizeof(double)));
    CK(cudaMemcpy(dAt, hAt, 16 * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBt, hBt, 16 * sizeof(double), cudaMemcpyHostToDevice));
    BK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, T, T, T, &d1, dAt, T, dBt, T,
                   &d0, dRt, T));
    if (ozaki2_dgemm(0, 0, T, T, T, dAt, T, dBt, T, dOt, T) != 0) {
      fprintf(stderr, "4x4 ozaki2_dgemm failed\n");
      return 1;
    }
    double hr[16], ho[16];
    CK(cudaMemcpy(hr, dRt, 16 * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ho, dOt, 16 * sizeof(double), cudaMemcpyDeviceToHost));
    const double e4 = max_abs_diff(hr, ho, 16);
    printf("4x4 max abs diff: %.6e\n", e4);
    if (e4 > tol4) {
      fprintf(stderr, "4x4 FAIL\n");
      return 1;
    }
  }

  // x GEMM shape
  BK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, NQ, Ncols, NQ, &d1, dA, NQ, dB,
                  NQ, &d0, dRef, NQ));
  if (ozaki2_dgemm(0, 0, NQ, Ncols, NQ, dA, NQ, dB, NQ, dOz, NQ) != 0) {
    fprintf(stderr, "ozaki2_dgemm failed\n");
    return 1;
  }
  std::vector<double> ref(MN), oz(MN);
  CK(cudaMemcpy(ref.data(), dRef, static_cast<size_t>(MN) * sizeof(double),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(oz.data(), dOz, static_cast<size_t>(MN) * sizeof(double),
                cudaMemcpyDeviceToHost));
  const double x_err = max_abs_diff(ref.data(), oz.data(), MN);
  printf("x GEMM max abs diff: %.6e\n", x_err);

  // y batched shape
  BK(cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, NQ, NQ, NQ, &d1, dB,
                               NQ, sp, dA, NQ, 0, &d0, dRef, NQ, sp, plane_batch));
  if (ozaki2_dgemm_strided_batched(0, 0, NQ, NQ, NQ, dB, NQ, sp, dA, NQ, 0, dOz,
                                   NQ, sp, plane_batch) != 0) {
    fprintf(stderr, "ozaki2_dgemm_strided_batched y failed\n");
    return 1;
  }
  CK(cudaMemcpy(ref.data(), dRef, static_cast<size_t>(MN) * sizeof(double),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(oz.data(), dOz, static_cast<size_t>(MN) * sizeof(double),
                cudaMemcpyDeviceToHost));
  const double y_err = max_abs_diff(ref.data(), oz.data(), MN);
  printf("y GEMM max abs diff: %.6e\n", y_err);

  // z batched shape
  BK(cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, NQ2, NQ, NQ, &d1, dB,
                               NQ2, MN, dA, NQ, 0, &d0, dRef, NQ2, MN, Ne));
  if (ozaki2_dgemm_strided_batched(0, 0, NQ2, NQ, NQ, dB, NQ2, MN, dA, NQ, 0, dOz,
                                   NQ2, MN, Ne) != 0) {
    fprintf(stderr, "ozaki2_dgemm_strided_batched z failed\n");
    return 1;
  }
  CK(cudaMemcpy(ref.data(), dRef, static_cast<size_t>(MN) * sizeof(double),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(oz.data(), dOz, static_cast<size_t>(MN) * sizeof(double),
                cudaMemcpyDeviceToHost));
  const double z_err = max_abs_diff(ref.data(), oz.data(), MN);
  printf("z GEMM max abs diff: %.6e\n", z_err);

  const double tol = 2.0e-2;
  printf("x GEMM max abs diff: %.6e (tol %.1e)\n", x_err, tol);
  if (x_err > tol || y_err > tol || z_err > tol) {
    fprintf(stderr, "FAIL: error exceeds %.1e\n", tol);
    return 1;
  }
  printf("PASS\n");

  ozaki2_finalize();
  cublasDestroy(h);
  return 0;
}
