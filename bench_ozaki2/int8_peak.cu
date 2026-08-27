// Achievable INT8 Tensor Core throughput on this GPU, as a function of K.
// Purpose: separate "the INT8 math is the limit" from "writing the INT32 C is
// the limit" in the Ozaki-II floor test.  M and N are held at the shape of the
// p=255 x GEMM (256 x 65536); only K varies.  A deep-K point and a large
// square point give the compute ceiling.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("cuda %s\n",cudaGetErrorString(e));exit(1);} } while(0)
#define BK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cublas %d\n",(int)s);exit(1);} } while(0)

static double run(cublasHandle_t h, int M, int N, int K,
                  signed char *A, signed char *B, int *C)
{
  const int one = 1, zero = 0;
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  auto call = [&]{ BK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                     &one, A, CUDA_R_8I, K, B, CUDA_R_8I, K,
                     &zero, C, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT)); };
  for (int i = 0; i < 5; ++i) call();
  CK(cudaDeviceSynchronize());
  std::vector<double> us;
  for (int i = 0; i < 15; ++i) {
    float ms; CK(cudaEventRecord(a)); call();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b)); us.push_back(ms * 1000.0);
  }
  std::sort(us.begin(), us.end());
  cudaEventDestroy(a); cudaEventDestroy(b);
  return us[us.size() / 2];
}

int main()
{
  cublasHandle_t h; BK(cublasCreate(&h));
  const int M = 256; const int N = 65536;
  signed char *A, *B; int *C;
  CK(cudaMalloc(&A, (size_t)16384 * 16384));
  CK(cudaMalloc(&B, (size_t)16384 * 65536));
  CK(cudaMalloc(&C, (size_t)16384 * 16384 * sizeof(int)));
  CK(cudaMemset(A, 1, (size_t)16384 * 16384));
  CK(cudaMemset(B, 1, (size_t)16384 * 65536));

  printf("# INT8 cublasGemmEx, M=%d N=%d, K swept\n", M, N);
  printf("#   K      us     TOP/s   C-write TB/s\n");
  for (int K : {128, 256, 512, 1024, 2048, 4096, 8192, 16384}) {
    double us = run(h, M, N, K, A, B, C);
    double top = 2.0 * M * (double)N * K / (us * 1e-6) / 1e12;
    double bw  = ((double)M * N * 4 + (double)K * N) / (us * 1e-6) / 1e12;
    printf("%6d %8.1f %9.1f %10.2f\n", K, us, top, bw);
  }
  printf("\n# large square INT8 (compute ceiling)\n");
  for (int n : {4096, 8192, 16384}) {
    double us = run(h, n, n, n, A, B, C);
    printf("  %5d^3 %8.1f us  %9.1f TOP/s\n", n, us, 2.0 * n * (double)n * n / (us * 1e-6) / 1e12);
  }
  return 0;
}
