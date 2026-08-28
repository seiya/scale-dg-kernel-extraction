#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

//- Defined in cuda_dg_kernels_tc.cu; the stream shared by the whole CUDA path.
extern cudaStream_t dg_cuda_stream;

static cublasHandle_t g_handle = nullptr;
static int g_inited = 0;
static void *g_workspace = nullptr;

// cuBLAS documents 8 GiB as the maximum workspace required by fixed-point
// FP64 emulation.  Keep that allocation for the lifetime of the handle so
// stage-level stream synchronizations cannot make cuBLAS allocate it again.
static constexpr size_t kEmulationWorkspaceBytes = size_t{8} << 30;

static int apply_emulation(int enable)
{
  if (!g_inited) {
    return 1;
  }

#if CUBLAS_VERSION >= 130002
  cublasStatus_t st;
  if (enable) {
    st = cublasSetEmulationStrategy(g_handle, CUBLAS_EMULATION_STRATEGY_EAGER);
    if (st != CUBLAS_STATUS_SUCCESS) {
      return static_cast<int>(st);
    }
  } else {
    st = cublasSetEmulationStrategy(g_handle, CUBLAS_EMULATION_STRATEGY_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
      return static_cast<int>(st);
    }
    st = cublasSetMathMode(g_handle, CUBLAS_DEFAULT_MATH);
    if (st != CUBLAS_STATUS_SUCCESS) {
      return static_cast<int>(st);
    }
  }
  return 0;
#else
  if (enable) {
    std::fprintf(stderr,
                 "cuBLAS FP64 emulation requires cuBLAS 13.0 update 2 or newer\n");
    return static_cast<int>(CUBLAS_STATUS_NOT_SUPPORTED);
  }
  return 0;
#endif
}

static int configure_handle()
{
  cublasStatus_t st = cublasSetStream(g_handle, dg_cuda_stream);
  if (st != CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(st);
  }
  if (g_workspace) {
    st = cublasSetWorkspace(g_handle, g_workspace, kEmulationWorkspaceBytes);
    if (st != CUBLAS_STATUS_SUCCESS) {
      return static_cast<int>(st);
    }
  }
  return 0;
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
  const int emulation_status = apply_emulation(emulate);
  if (emulation_status != 0) {
    return emulation_status;
  }

  if (emulate && !g_workspace) {
    const cudaError_t st = cudaMalloc(&g_workspace, kEmulationWorkspaceBytes);
    if (st != cudaSuccess) {
      return static_cast<int>(st);
    }
  }
  return configure_handle();
}

extern "C" int cublas_gemm_set_emulation(int emulate)
{
  return apply_emulation(emulate);
}

extern "C" int cublas_gemm_finalize(void)
{
  int result = 0;
  if (g_inited) {
    const cublasStatus_t st = cublasDestroy(g_handle);
    g_handle = nullptr;
    g_inited = 0;
    result = static_cast<int>(st);
  }
  if (g_workspace) {
    const cudaError_t st = cudaFree(g_workspace);
    g_workspace = nullptr;
    if (result == 0) {
      result = static_cast<int>(st);
    }
  }
  return result;
}

extern "C" int cublas_dgemm_impl(int transa, int transb, int m, int n, int k,
                                 double alpha, const double *A, int lda,
                                 const double *B, int ldb, double beta,
                                 double *C, int ldc)
{
  const cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
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
  return static_cast<int>(cublasDgemmStridedBatched(
      g_handle, opA, opB, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB,
      &beta, C, ldc, strideC, batch));
}

extern "C" int cublas_gemm_ex_impl(int transa, int transb, int m, int n, int k,
                                   const int *alpha, const signed char *A,
                                   int lda, const signed char *B, int ldb,
                                   const int *beta, int *C, int ldc)
{
  if (!g_inited) {
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);
  }
  const cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  return static_cast<int>(
      cublasGemmEx(g_handle, opA, opB, m, n, k, alpha, A, CUDA_R_8I, lda, B,
                   CUDA_R_8I, ldb, beta, C, CUDA_R_32I, ldc, CUBLAS_COMPUTE_32I,
                   CUBLAS_GEMM_DEFAULT));
}

extern "C" int cublas_gemm_strided_batched_ex_impl(
    int transa, int transb, int m, int n, int k, const int *alpha,
    const signed char *A, int lda, long long strideA, const signed char *B,
    int ldb, long long strideB, const int *beta, int *C, int ldc,
    long long strideC, int batch)
{
  if (!g_inited) {
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);
  }
  const cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  return static_cast<int>(
      cublasGemmStridedBatchedEx(g_handle, opA, opB, m, n, k, alpha, A,
                                 CUDA_R_8I, lda, strideA, B, CUDA_R_8I, ldb,
                                 strideB, beta, C, CUDA_R_32I, ldc, strideC,
                                 batch, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
}
