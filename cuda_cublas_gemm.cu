#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

//- Defined in cuda_dg_kernels_tc.cu; the stream shared by the whole CUDA path.
extern cudaStream_t dg_cuda_stream;

static cublasHandle_t g_handle = nullptr;
static int g_inited = 0;

static int apply_emulation(int enable)
{
  if (!g_inited) {
    return 1;
  }

#if defined(CUBLAS_EMULATION_STRATEGY_EAGER)
  cublasStatus_t st;
  if (enable) {
    st = cublasSetEmulationStrategy(g_handle, CUBLAS_EMULATION_STRATEGY_EAGER);
    if (st != CUBLAS_STATUS_SUCCESS) {
      return static_cast<int>(st);
    }
  } else {
#if defined(CUBLAS_EMULATION_STRATEGY_DEFAULT)
    st = cublasSetEmulationStrategy(g_handle, CUBLAS_EMULATION_STRATEGY_DEFAULT);
#else
    st = cublasSetEmulationStrategy(g_handle, CUBLAS_EMULATION_STRATEGY_PERFORMANT);
#endif
    if (st != CUBLAS_STATUS_SUCCESS) {
      return static_cast<int>(st);
    }
    st = cublasSetMathMode(g_handle, CUBLAS_PEDANTIC_MATH);
    if (st != CUBLAS_STATUS_SUCCESS) {
      return static_cast<int>(st);
    }
  }
  return 0;
#else
  if (enable) {
    std::fprintf(stderr,
                 "cuBLAS floating-point emulation APIs are unavailable; "
                 "native FP64 GEMM will be used\n");
  }
  return 0;
#endif
}

extern "C" int cublas_gemm_init(int emulate)
{
  if (!emulate) {
    setenv("CUBLAS_EMULATE_DOUBLE_PRECISION", "0", 1);
    setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "0", 1);
  } else {
    setenv("CUBLAS_EMULATION_STRATEGY", "eager", 1);
    setenv("CUBLAS_EMULATE_DOUBLE_PRECISION", "1", 1);
  }

  if (!g_inited) {
    const cublasStatus_t st = cublasCreate(&g_handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
      return static_cast<int>(st);
    }
    g_inited = 1;
  }
  return apply_emulation(emulate);
}

extern "C" int cublas_gemm_set_emulation(int emulate)
{
  return apply_emulation(emulate);
}

extern "C" int cublas_gemm_finalize(void)
{
  if (g_inited) {
    const cublasStatus_t st = cublasDestroy(g_handle);
    g_handle = nullptr;
    g_inited = 0;
    return static_cast<int>(st);
  }
  return 0;
}

extern "C" int cublas_dgemm_impl(int transa, int transb, int m, int n, int k,
                                 double alpha, const double *A, int lda,
                                 const double *B, int ldb, double beta,
                                 double *C, int ldc)
{
  const cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSetStream(g_handle, dg_cuda_stream);
  return static_cast<int>(cublasDgemm(g_handle, opA, opB, m, n, k, &alpha, A,
                                      lda, B, ldb, &beta, C, ldc));
}

extern "C" int cublas_dgemm_strided_batched_impl(
    int transa, int transb, int m, int n, int k, double alpha, const double *A,
    int lda, long long strideA, const double *B, int ldb, long long strideB,
    double beta, double *C, int ldc, long long strideC, int batch)
{
  const cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSetStream(g_handle, dg_cuda_stream);
  return static_cast<int>(cublasDgemmStridedBatched(
      g_handle, opA, opB, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB,
      &beta, C, ldc, strideC, batch));
}
