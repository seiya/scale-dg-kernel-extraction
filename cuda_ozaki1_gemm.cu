// Ozaki Scheme I volume GEMM emulation: slice both operands, accumulate s^2 INT8 GEMMs.
// No CRT — each slice pair contributes scale_a[i]*scale_b[j]*INT32_GEMM to FP64 C.

#include "cuda_ozaki1_gemm.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" int cublas_gemm_ex_impl(int transa, int transb, int m, int n, int k,
                                   const int *alpha, const signed char *A,
                                   int lda, const signed char *B, int ldb,
                                   const int *beta, int *C, int ldc);

extern "C" int cublas_gemm_strided_batched_ex_impl(
    int transa, int transb, int m, int n, int k, const int *alpha,
    const signed char *A, int lda, long long strideA, const signed char *B,
    int ldb, long long strideB, const int *beta, int *C, int ldc,
    long long strideC, int batch);

extern cudaStream_t dg_cuda_stream;

static constexpr int kMaxSlices = 16;
static constexpr int kMinSlices = 2;
static constexpr int kDefaultSlices = 8;

struct Ozaki1State {
  int slice_count = 0;
  int inited = 0;
};

static Ozaki1State g_state;

struct Ozaki1Workspace {
  int Nq = 0;
  int Ne = 0;
  int Np = 0;
  int max_m = 0;
  long long mn = 0;
  int max_batch = 0;
  signed char *iA = nullptr;
  signed char *iB = nullptr;
  double *scale_a = nullptr;
  double *scale_b = nullptr;
  int *prod = nullptr;
  double *res_a = nullptr;
  double *res_b = nullptr;
  int slices_a = 0;
  int slices_b = 0;
};

static Ozaki1Workspace g_ws;

static constexpr double kSliceThreshold = 1.0;

static int check_cuda(cudaError_t err, const char *what)
{
  if (err != cudaSuccess) {
    std::fprintf(stderr, "ozaki1 CUDA error %s: %s\n", what,
                 cudaGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

static int sync_stream()
{
  return check_cuda(cudaStreamSynchronize(dg_cuda_stream), "sync");
}

__global__ void ozaki1_row_max_kernel(const double *__restrict__ mat,
                                      double *__restrict__ scale, int rows,
                                      int cols, int lda)
{
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    double maxv = 0.0;
    for (int c = 0; c < cols; ++c) {
      maxv = fmax(maxv, fabs(mat[row + static_cast<long long>(c) * lda]));
    }
    scale[row] = (maxv > 0.0) ? maxv / 127.0 : 0.0;
  }
}

__global__ void ozaki1_col_max_kernel(const double *__restrict__ mat,
                                      double *__restrict__ scale, int rows,
                                      int cols, int ldb)
{
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
       col += blockDim.x * gridDim.x) {
    double maxv = 0.0;
    for (int r = 0; r < rows; ++r) {
      maxv = fmax(maxv, fabs(mat[r + static_cast<long long>(col) * ldb]));
    }
    scale[col] = (maxv > 0.0) ? maxv / 127.0 : 0.0;
  }
}

__global__ void ozaki1_pack_a_tn_kernel(const double *__restrict__ mat,
                                        signed char *__restrict__ out,
                                        const double *__restrict__ scale, int rows,
                                        int cols, int lda)
{
  const long long total = static_cast<long long>(rows) * cols;
  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int col = static_cast<int>(idx / rows);
    const int row = static_cast<int>(idx % rows);
    const double denom = scale[row];
    const double raw = mat[row + static_cast<long long>(col) * lda];
    const double v = (denom > 0.0) ? raw / denom : 0.0;
    int q = static_cast<int>(lrint(v));
    q = max(-127, min(127, q));
    out[col + static_cast<long long>(row) * cols] = static_cast<signed char>(q);
  }
}

__global__ void ozaki1_pack_b_kernel(const double *__restrict__ mat,
                                     signed char *__restrict__ out,
                                     const double *__restrict__ scale, int rows,
                                     int cols, int ldb)
{
  const long long total = static_cast<long long>(rows) * cols;
  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int col = static_cast<int>(idx / rows);
    const int row = static_cast<int>(idx % rows);
    const double denom = scale[col];
    const double raw = mat[row + static_cast<long long>(col) * ldb];
    const double v = (denom > 0.0) ? raw / denom : 0.0;
    int q = static_cast<int>(lrint(v));
    q = max(-127, min(127, q));
    out[row + static_cast<long long>(col) * rows] = static_cast<signed char>(q);
  }
}

__global__ void ozaki1_residual_a_tn_kernel(const double *__restrict__ mat,
                                            const signed char *__restrict__ packed,
                                            const double *__restrict__ scale,
                                            double *__restrict__ res, int rows,
                                            int cols, int lda)
{
  const long long total = static_cast<long long>(rows) * cols;
  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int col = static_cast<int>(idx / rows);
    const int row = static_cast<int>(idx % rows);
    const double q = static_cast<double>(
        packed[col + static_cast<long long>(row) * cols]);
    res[row + static_cast<long long>(col) * lda] =
        mat[row + static_cast<long long>(col) * lda] - scale[row] * q;
  }
}

__global__ void ozaki1_residual_b_kernel(const double *__restrict__ mat,
                                         const signed char *__restrict__ packed,
                                         const double *__restrict__ scale,
                                         double *__restrict__ res, int rows,
                                         int cols, int ldb)
{
  const long long total = static_cast<long long>(rows) * cols;
  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int col = static_cast<int>(idx / rows);
    const int row = static_cast<int>(idx % rows);
    const double q = static_cast<double>(
        packed[row + static_cast<long long>(col) * rows]);
    res[row + static_cast<long long>(col) * ldb] =
        mat[row + static_cast<long long>(col) * ldb] - scale[col] * q;
  }
}

__global__ void ozaki1_row_max_batched_kernel(const double *__restrict__ mat,
                                              double *__restrict__ scale, int rows,
                                              int cols, int lda, long long stride,
                                              int batch)
{
  const long long total_rows =
      static_cast<long long>(rows) * static_cast<long long>(batch);
  for (long long g = blockIdx.x * blockDim.x + threadIdx.x; g < total_rows;
       g += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int b = static_cast<int>(g / rows);
    const int row = static_cast<int>(g % rows);
    const double *slice = mat + static_cast<long long>(b) * stride;
    double *sc = scale + static_cast<long long>(b) * rows;
    double maxv = 0.0;
    for (int c = 0; c < cols; ++c) {
      maxv = fmax(maxv, fabs(slice[row + static_cast<long long>(c) * lda]));
    }
    sc[row] = (maxv > 0.0) ? maxv / 127.0 : 0.0;
  }
}

__global__ void ozaki1_pack_a_batched_tn_kernel(const double *__restrict__ mat,
                                               signed char *__restrict__ out,
                                               const double *__restrict__ scale,
                                               int rows, int cols, int lda,
                                               long long stride, int batch)
{
  const long long plane = static_cast<long long>(rows) * cols;
  const long long total = plane * static_cast<long long>(batch);
  for (long long g = blockIdx.x * blockDim.x + threadIdx.x; g < total;
       g += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int b = static_cast<int>(g / plane);
    const long long idx = g - static_cast<long long>(b) * plane;
    const int col = static_cast<int>(idx / rows);
    const int row = static_cast<int>(idx % rows);
    const double *slice = mat + static_cast<long long>(b) * stride;
    signed char *oslice = out + static_cast<long long>(b) * stride;
    const double *sc = scale + static_cast<long long>(b) * rows;
    const double denom = sc[row];
    const double raw = slice[row + static_cast<long long>(col) * lda];
    const double v = (denom > 0.0) ? raw / denom : 0.0;
    int q = static_cast<int>(lrint(v));
    q = max(-127, min(127, q));
    oslice[col + static_cast<long long>(row) * cols] = static_cast<signed char>(q);
  }
}

__global__ void ozaki1_residual_a_batched_tn_kernel(
    const double *__restrict__ mat, const signed char *__restrict__ packed,
    const double *__restrict__ scale, double *__restrict__ res, int rows, int cols,
    int lda, long long stride, int batch)
{
  const long long plane = static_cast<long long>(rows) * cols;
  const long long total = plane * static_cast<long long>(batch);
  for (long long g = blockIdx.x * blockDim.x + threadIdx.x; g < total;
       g += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int b = static_cast<int>(g / plane);
    const long long idx = g - static_cast<long long>(b) * plane;
    const int col = static_cast<int>(idx / rows);
    const int row = static_cast<int>(idx % rows);
    const double *slice = mat + static_cast<long long>(b) * stride;
    const signed char *pslice = packed + static_cast<long long>(b) * stride;
    double *rslice = res + static_cast<long long>(b) * stride;
    const double *sc = scale + static_cast<long long>(b) * rows;
    const double q = static_cast<double>(
        pslice[col + static_cast<long long>(row) * cols]);
    rslice[row + static_cast<long long>(col) * lda] =
        slice[row + static_cast<long long>(col) * lda] - sc[row] * q;
  }
}

__global__ void ozaki1_recon_kernel(double *__restrict__ C,
                                    const int *__restrict__ prod,
                                    const double *__restrict__ scale_a,
                                    const double *__restrict__ scale_b,
                                    int m, int n, int ldc, int accumulate)
{
  const long long mn = static_cast<long long>(m) * n;
  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < mn;
       idx += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int col = static_cast<int>(idx / m);
    const int row = static_cast<int>(idx % m);
    const double val =
        scale_a[row] * static_cast<double>(prod[idx]) * scale_b[col];
    if (accumulate) {
      C[row + static_cast<long long>(col) * ldc] += val;
    } else {
      C[row + static_cast<long long>(col) * ldc] = val;
    }
  }
}

__global__ void ozaki1_recon_batched_kernel(double *__restrict__ C,
                                            const int *__restrict__ prod,
                                            const double *__restrict__ scale_a,
                                            const double *__restrict__ scale_b,
                                            int m, int n, int ldc,
                                            long long stride_c, int batch,
                                            int accumulate)
{
  const long long mn = static_cast<long long>(m) * n;
  const long long total = mn * static_cast<long long>(batch);
  for (long long g = blockIdx.x * blockDim.x + threadIdx.x; g < total;
       g += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int b = static_cast<int>(g / mn);
    const long long idx = g - static_cast<long long>(b) * mn;
    const int col = static_cast<int>(idx / m);
    const int row = static_cast<int>(idx % m);
    const double sa = scale_a[static_cast<long long>(b) * m + row];
    const double val = sa * static_cast<double>(prod[g]) * scale_b[col];
    const long long off = row + static_cast<long long>(col) * ldc + b * stride_c;
    if (accumulate) {
      C[off] += val;
    } else {
      C[off] = val;
    }
  }
}

static bool scales_need_next_slice(const double *scale, int count)
{
  std::vector<double> hscale(static_cast<size_t>(count));
  if (check_cuda(
          cudaMemcpy(hscale.data(), scale, hscale.size() * sizeof(double),
                     cudaMemcpyDeviceToHost),
          "slice threshold")) {
    return false;
  }
  double mx = 0.0;
  for (const double s : hscale) {
    mx = std::fmax(mx, s * 127.0);
  }
  return mx > kSliceThreshold;
}

static int launch_recon(double *C, const int *prod, const double *scale_a,
                        const double *scale_b, int m, int n, int ldc,
                        int accumulate)
{
  const long long mn = static_cast<long long>(m) * n;
  const int blocks = static_cast<int>((mn + 255) / 256);
  ozaki1_recon_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(
      C, prod, scale_a, scale_b, m, n, ldc, accumulate);
  return check_cuda(cudaGetLastError(), "recon");
}

static int launch_recon_batched(double *C, const int *prod,
                                const double *scale_a, const double *scale_b,
                                int m, int n, int ldc, long long stride_c,
                                int batch, int accumulate)
{
  const long long total = static_cast<long long>(m) * n * batch;
  const int blocks = static_cast<int>((total + 255) / 256);
  ozaki1_recon_batched_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(
      C, prod, scale_a, scale_b, m, n, ldc, stride_c, batch, accumulate);
  return check_cuda(cudaGetLastError(), "recon_batched");
}

static int decompose_a_tn(const double *A, int m, int k, int lda, int batch,
                          long long stride, int *slices_used)
{
  const long long plane = static_cast<long long>(m) * k;
  const double *src = A;
  int used = 0;
  for (int si = 0; si < g_state.slice_count; ++si) {
    double *sc =
        g_ws.scale_a + static_cast<long long>(si) * g_ws.max_m * g_ws.max_batch;
    signed char *pack = g_ws.iA + static_cast<long long>(si) * plane * batch;
    if (batch == 1 && stride == 0) {
      const int row_blocks = (m + 255) / 256;
      ozaki1_row_max_kernel<<<row_blocks, 256, 0, dg_cuda_stream>>>(
          src, sc, m, k, lda);
      int err = check_cuda(cudaGetLastError(), "row_max");
      if (err) return err;
      const int pack_blocks = static_cast<int>((plane + 255) / 256);
      ozaki1_pack_a_tn_kernel<<<pack_blocks, 256, 0, dg_cuda_stream>>>(
          src, pack, sc, m, k, lda);
      err = check_cuda(cudaGetLastError(), "pack_a");
      if (err) return err;
    } else {
      const long long total_rows =
          static_cast<long long>(m) * static_cast<long long>(batch);
      const int row_blocks = static_cast<int>((total_rows + 255) / 256);
      ozaki1_row_max_batched_kernel<<<row_blocks, 256, 0, dg_cuda_stream>>>(
          src, sc, m, k, lda, stride, batch);
      int err = check_cuda(cudaGetLastError(), "row_max_batched");
      if (err) return err;
      const long long total_pack = plane * static_cast<long long>(batch);
      const int pack_blocks = static_cast<int>((total_pack + 255) / 256);
      ozaki1_pack_a_batched_tn_kernel<<<pack_blocks, 256, 0, dg_cuda_stream>>>(
          src, pack, sc, m, k, lda, stride, batch);
      err = check_cuda(cudaGetLastError(), "pack_a_batched");
      if (err) return err;
    }
    used = si + 1;
    if (si + 1 >= g_state.slice_count) break;
    if (!scales_need_next_slice(sc, m * batch)) break;
    if (batch == 1 && stride == 0) {
      const int blocks = static_cast<int>((plane + 255) / 256);
      ozaki1_residual_a_tn_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(
          src, pack, sc, g_ws.res_a, m, k, lda);
      int err = check_cuda(cudaGetLastError(), "residual_a");
      if (err) return err;
    } else {
      const long long total_pack = plane * static_cast<long long>(batch);
      const int pack_blocks = static_cast<int>((total_pack + 255) / 256);
      ozaki1_residual_a_batched_tn_kernel<<<pack_blocks, 256, 0, dg_cuda_stream>>>(
          src, pack, sc, g_ws.res_a, m, k, lda, stride, batch);
      int err = check_cuda(cudaGetLastError(), "residual_a_batched");
      if (err) return err;
    }
    src = g_ws.res_a;
  }
  *slices_used = used;
  return 0;
}

static int decompose_b_nn(const double *B, int k, int n, int ldb, int *slices_used)
{
  const long long plane = static_cast<long long>(k) * n;
  const double *src = B;
  int used = 0;
  for (int sj = 0; sj < g_state.slice_count; ++sj) {
    double *sc = g_ws.scale_b + static_cast<long long>(sj) * n;
    signed char *pack = g_ws.iB + static_cast<long long>(sj) * plane;
    const int col_blocks = (n + 255) / 256;
    ozaki1_col_max_kernel<<<col_blocks, 256, 0, dg_cuda_stream>>>(
        src, sc, k, n, ldb);
    int err = check_cuda(cudaGetLastError(), "col_max");
    if (err) return err;
    const int pack_blocks = static_cast<int>((plane + 255) / 256);
    ozaki1_pack_b_kernel<<<pack_blocks, 256, 0, dg_cuda_stream>>>(
        src, pack, sc, k, n, ldb);
    err = check_cuda(cudaGetLastError(), "pack_b");
    if (err) return err;
    used = sj + 1;
    if (sj + 1 >= g_state.slice_count) break;
    if (!scales_need_next_slice(sc, n)) break;
    ozaki1_residual_b_kernel<<<pack_blocks, 256, 0, dg_cuda_stream>>>(
        src, pack, sc, g_ws.res_b, k, n, ldb);
    err = check_cuda(cudaGetLastError(), "residual_b");
    if (err) return err;
    src = g_ws.res_b;
  }
  *slices_used = used;
  return 0;
}

static int run_slice_pairs(int m, int n, int k, double *C, int ldc, int sa, int sb,
                           long long stride_c, int batch, bool batched)
{
  const long long plane_a = static_cast<long long>(k) * m;
  const long long plane_b = static_cast<long long>(k) * n;
  const int i_one = 1;
  const int i_zero = 0;
  bool accumulate = false;
  int err = 0;

  for (int si = 0; si < sa; ++si) {
    for (int sj = 0; sj < sb; ++sj) {
      const signed char *iAp =
          g_ws.iA + static_cast<long long>(si) * plane_a * batch;
      const signed char *iBp = g_ws.iB + static_cast<long long>(sj) * plane_b;
      const double *scA = g_ws.scale_a +
                          static_cast<long long>(si) * g_ws.max_m * g_ws.max_batch;
      const double *scB = g_ws.scale_b + static_cast<long long>(sj) * n;

      if (!batched) {
        const int istat =
            cublas_gemm_ex_impl(1, 0, m, n, k, &i_one, iAp, k, iBp, k, &i_zero,
                                g_ws.prod, m);
        if (istat != 0) {
          std::fprintf(stderr, "ozaki1: cublasGemmEx failed %d\n", istat);
          return istat;
        }
        err = launch_recon(C, g_ws.prod, scA, scB, m, n, ldc, accumulate);
      } else {
        const int istat = cublas_gemm_strided_batched_ex_impl(
            1, 0, m, n, k, &i_one, iAp, k, plane_a, iBp, k, 0, &i_zero,
            g_ws.prod, m, stride_c, batch);
        if (istat != 0) {
          std::fprintf(stderr, "ozaki1: cublasGemmEx strided failed %d\n", istat);
          return istat;
        }
        err = launch_recon_batched(C, g_ws.prod, scA, scB, m, n, ldc, stride_c,
                                   batch, accumulate);
      }
      if (err) return err;
      accumulate = true;
    }
  }
  return 0;
}

extern "C" int ozaki1_init(int slice_count)
{
  if (slice_count < kMinSlices || slice_count > kMaxSlices) {
    std::fprintf(stderr,
                 "ozaki1_init: slice_count must be in [%d, %d], got %d\n",
                 kMinSlices, kMaxSlices, slice_count);
    return -1;
  }
  g_state.slice_count = slice_count;
  g_state.inited = 1;
  return 0;
}

extern "C" int ozaki1_finalize(void)
{
  ozaki1_free_workspace();
  g_state = Ozaki1State{};
  return 0;
}

extern "C" int ozaki1_alloc_workspace(int Nq, int Ne, int Np)
{
  if (!g_state.inited) {
    const int rc = ozaki1_init(kDefaultSlices);
    if (rc != 0) return rc;
  }
  ozaki1_free_workspace();

  g_ws.Nq = Nq;
  g_ws.Ne = Ne;
  g_ws.Np = Np;
  g_ws.max_m = Nq * Nq;
  g_ws.mn = static_cast<long long>(Np) * Ne;
  g_ws.max_batch = std::max(Nq * Ne, Ne);

  const long long k_rows = static_cast<long long>(Nq) * Nq;
  const long long mn = g_ws.mn;
  const int slices = g_state.slice_count;
  const int max_batch = g_ws.max_batch;
  const int max_m = g_ws.max_m;

  const size_t iA_bytes =
      static_cast<size_t>(slices) * k_rows * static_cast<size_t>(max_batch);
  const size_t iB_bytes = static_cast<size_t>(slices) * static_cast<size_t>(mn);
  const size_t scale_a_bytes = static_cast<size_t>(slices) *
                               static_cast<size_t>(max_m) *
                               static_cast<size_t>(max_batch) * sizeof(double);
  const size_t scale_b_bytes =
      static_cast<size_t>(slices) * static_cast<size_t>(mn) * sizeof(double);
  const size_t prod_bytes = static_cast<size_t>(mn) * sizeof(int);
  const size_t res_bytes = static_cast<size_t>(mn) * sizeof(double);

  int err = 0;
  err = check_cuda(cudaMalloc(&g_ws.iA, iA_bytes), "malloc iA");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.iB, iB_bytes), "malloc iB");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.scale_a, scale_a_bytes), "malloc scale_a");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.scale_b, scale_b_bytes), "malloc scale_b");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.prod, prod_bytes), "malloc prod");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.res_a, res_bytes), "malloc res_a");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.res_b, res_bytes), "malloc res_b");
  if (err) return err;

  std::printf("ozaki1 workspace: Nq=%d Ne=%d Np=%d slices=%d iB=%.1f MB\n", Nq, Ne,
              Np, slices, iB_bytes / 1.0e6);
  return 0;
}

extern "C" void ozaki1_free_workspace(void)
{
  if (g_ws.iA) cudaFree(g_ws.iA);
  if (g_ws.iB) cudaFree(g_ws.iB);
  if (g_ws.scale_a) cudaFree(g_ws.scale_a);
  if (g_ws.scale_b) cudaFree(g_ws.scale_b);
  if (g_ws.prod) cudaFree(g_ws.prod);
  if (g_ws.res_a) cudaFree(g_ws.res_a);
  if (g_ws.res_b) cudaFree(g_ws.res_b);
  g_ws = Ozaki1Workspace{};
}

static int ensure_workspace(int m, int n, int k, int batch)
{
  if (!g_ws.prod) {
    std::fprintf(stderr, "ozaki1: workspace not allocated\n");
    return -1;
  }
  if (static_cast<long long>(m) * n * static_cast<long long>(batch) > g_ws.mn) {
    std::fprintf(stderr, "ozaki1: output exceeds workspace\n");
    return -1;
  }
  if (batch > g_ws.max_batch) {
    std::fprintf(stderr, "ozaki1: batch=%d exceeds max_batch=%d\n", batch,
                 g_ws.max_batch);
    return -1;
  }
  return 0;
}

extern "C" int ozaki1_dgemm(int transa, int transb, int m, int n, int k,
                            const double *A, int lda, const double *B, int ldb,
                            double *C, int ldc)
{
  if (!g_state.inited) return -1;
  if (transa != 0 || transb != 0) {
    std::fprintf(stderr, "ozaki1_dgemm: only NN supported\n");
    return -1;
  }
  if (ensure_workspace(m, n, k, 1) != 0) return -1;

  int sa = 0, sb = 0;
  int err = decompose_b_nn(B, k, n, ldb, &sb);
  if (err) return err;
  err = decompose_a_tn(A, m, k, lda, 1, 0, &sa);
  if (err) return err;
  g_ws.slices_a = sa;
  g_ws.slices_b = sb;
  err = run_slice_pairs(m, n, k, C, ldc, sa, sb, 0, 1, false);
  if (err) return err;
  return sync_stream();
}

extern "C" int ozaki1_dgemm_strided_batched(int transa, int transb, int m, int n,
                                            int k, const double *A, int lda,
                                            long long strideA, const double *B,
                                            int ldb, long long strideB,
                                            double *C, int ldc,
                                            long long strideC, int batch)
{
  if (!g_state.inited) return -1;
  if (transa != 0 || transb != 0) {
    std::fprintf(stderr, "ozaki1_dgemm_strided_batched: only NN supported\n");
    return -1;
  }
  if (ensure_workspace(m, n, k, batch) != 0) return -1;

  int sa = 0, sb = 0;
  int err = 0;
  if (strideB == 0) {
    err = decompose_b_nn(B, k, n, ldb, &sb);
    if (err) return err;
  } else {
    std::fprintf(stderr, "ozaki1: strided B decomposition not implemented\n");
    return -1;
  }

  if (strideA > 0) {
    err = decompose_a_tn(A, m, k, lda, batch, strideA, &sa);
  } else {
    err = decompose_a_tn(A, m, k, lda, 1, 0, &sa);
  }
  if (err) return err;

  g_ws.slices_a = sa;
  g_ws.slices_b = sb;
  err = run_slice_pairs(m, n, k, C, ldc, sa, sb, strideC, batch, true);
  if (err) return err;
  return sync_stream();
}
