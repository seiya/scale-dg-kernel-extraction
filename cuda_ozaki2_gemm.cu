// Ozaki Scheme II volume GEMM emulation (arXiv:2504.08009).
// Diagonal scaling, s modular INT8 GEMMs via cuBLAS, Garner CRT, inverse scale.

#include "cuda_ozaki2_gemm.h"

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

extern "C" int cublas_dgemm_impl(int transa, int transb, int m, int n, int k,
                                 double alpha, const double *A, int lda,
                                 const double *B, int ldb, double beta,
                                 double *C, int ldc);

extern "C" int cublas_dgemm_strided_batched_impl(
    int transa, int transb, int m, int n, int k, double alpha, const double *A,
    int lda, long long strideA, const double *B, int ldb, long long strideB,
    double beta, double *C, int ldc, long long strideC, int batch);

extern cudaStream_t dg_cuda_stream;

static constexpr int kMaxModuli = 20;
static constexpr int kMinModuli = 2;
// First 7 pool entries have log2(product) ≈ 55.7, matching cuBLAS FIXED 55.
static constexpr int kDefaultModuli = 7;

// INT8 moduli table aligned with RIKEN-RCCS/GEMMul8 (src/oz2/common/table.hpp).
// Not all entries are prime; they are pairwise coprime CRT moduli.
static const int kModuliPool[] = {
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199,
    197, 193, 191, 181, 179, 173};

static constexpr int kMaxASlices = 4;

struct Ozaki2State {
  int moduli_count = 0;
  int moduli[kMaxModuli] = {};
  int64_t garner_inv[kMaxModuli] = {};
  int fixed = 1;
  int inited = 0;
};

static Ozaki2State g_state;

struct Ozaki2Workspace {
  int Nq = 0;
  int Ne = 0;
  int Np = 0;
  int nq2 = 0;
  long long mn = 0;
  long long k_max = 0;
  int max_batch = 0;
  signed char *iA = nullptr;
  signed char *iB = nullptr;
  double *scale_a = nullptr;
  double *scale_b = nullptr;
  int *residues = nullptr;
  int *prod = nullptr;
  double *res_a = nullptr;
  size_t iA_bytes = 0;
  size_t iB_bytes = 0;
  size_t res_a_bytes = 0;
};

static Ozaki2Workspace g_ws;

static int launch_crt(double *C, const int *residues, int m, int n, int ldc,
                      int k_dim, int accumulate);
static int launch_crt_batched(double *C, const int *residues, int m, int n,
                              int ldc, long long stride_c, int batch, int k_dim,
                              int accumulate);
static int launch_split_moduli(long long n);

static int64_t mod_inv(int64_t a, int64_t m)
{
  int64_t t = 0, newt = 1;
  int64_t r = m, newr = a % m;
  if (newr < 0) newr += m;
  while (newr != 0) {
    const int64_t q = r / newr;
    const int64_t tmp_t = t - q * newt;
    t = newt;
    newt = tmp_t;
    const int64_t tmp_r = r - q * newr;
    r = newr;
    newr = tmp_r;
  }
  if (r > 1) return 0;
  if (t < 0) t += m;
  return t;
}

static void precompute_garner_inverses(Ozaki2State &st)
{
  unsigned __int128 cum = static_cast<unsigned __int128>(st.moduli[0]);
  for (int i = 1; i < st.moduli_count; ++i) {
    const int64_t pi = st.moduli[i];
    st.garner_inv[i - 1] = mod_inv(static_cast<int64_t>(cum % pi), pi);
    cum *= static_cast<unsigned __int128>(st.moduli[i]);
  }
}

static int check_cuda(cudaError_t err, const char *what)
{
  if (err != cudaSuccess) {
    std::fprintf(stderr, "ozaki2 CUDA error %s: %s\n", what,
                 cudaGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

static constexpr double kSecondSliceThreshold = 1.0;

static bool matrix_needs_second_slice(int rows, int batch)
{
  std::vector<double> hscale(static_cast<size_t>(rows) * static_cast<size_t>(batch));
  if (check_cuda(
          cudaMemcpy(hscale.data(), g_ws.scale_a,
                     hscale.size() * sizeof(double), cudaMemcpyDeviceToHost),
          "slice threshold memcpy")) {
    return false;
  }
  double mx = 0.0;
  for (const double s : hscale) {
    mx = std::fmax(mx, s * 127.0);
  }
  return mx > kSecondSliceThreshold;
}

static int sync_ozaki2_stream()
{
  return check_cuda(cudaStreamSynchronize(dg_cuda_stream), "sync");
}

static int run_int8_gemm_crt(int m, int n, int k, double *C, int ldc, int accumulate)
{
  const long long mn = static_cast<long long>(m) * n;
  const int i_one = 1;
  const int i_zero = 0;
  const int istat =
      cublas_gemm_ex_impl(1, 0, m, n, k, &i_one, g_ws.iA, k, g_ws.iB, k,
                          &i_zero, g_ws.prod, m);
  if (istat != 0) {
    std::fprintf(stderr, "ozaki2: cublasGemmEx failed %d\n", istat);
    return istat;
  }
  int err = launch_split_moduli(mn);
  if (err) return err;
  return launch_crt(C, g_ws.residues, m, n, ldc, k, accumulate);
}

static int run_int8_strided_gemm_crt(int m, int n, int k, double *C, int ldc,
                                     long long strideA, long long strideB,
                                     long long strideC, int batch,
                                     int accumulate)
{
  const long long mn = static_cast<long long>(m) * n;
  const long long total_out = mn * static_cast<long long>(batch);
  const int i_one = 1;
  const int i_zero = 0;
  const int istat = cublas_gemm_strided_batched_ex_impl(
      1, 0, m, n, k, &i_one, g_ws.iA, k, strideA, g_ws.iB, k, strideB, &i_zero,
      g_ws.prod, m, strideC, batch);
  if (istat != 0) {
    std::fprintf(stderr, "ozaki2: cublasGemmEx strided failed %d\n", istat);
    return istat;
  }
  int err = launch_split_moduli(total_out);
  if (err) return err;
  if (batch == 1 && strideC == mn) {
    return launch_crt(C, g_ws.residues, m, n, ldc, k, accumulate);
  }
  return launch_crt_batched(C, g_ws.residues, m, n, ldc, strideC, batch, k,
                            accumulate);
}

static void maybe_check_dgemm(const char *tag, int m, int n, int k,
                              const double *A, int lda, const double *B,
                              int ldb, const double *C, int ldc)
{
  if (!std::getenv("OZAKI2_CHECK")) {
    return;
  }
  const long long mn = static_cast<long long>(m) * n;
  double *ref = nullptr;
  if (check_cuda(cudaMalloc(&ref, static_cast<size_t>(mn) * sizeof(double)),
                 "check malloc")) {
    return;
  }
  const int istat =
      cublas_dgemm_impl(0, 0, m, n, k, 1.0, A, lda, B, ldb, 0.0, ref, ldc);
  if (istat != 0) {
    std::fprintf(stderr, "ozaki2 check %s: cublas ref failed %d\n", tag, istat);
    cudaFree(ref);
    return;
  }
  std::vector<double> href(static_cast<size_t>(mn)), ho(static_cast<size_t>(mn));
  cudaMemcpy(href.data(), ref, static_cast<size_t>(mn) * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(ho.data(), C, static_cast<size_t>(mn) * sizeof(double),
             cudaMemcpyDeviceToHost);
  double mx = 0.0;
  double maxA = 0.0;
  double maxB = 0.0;
  const long long mk = static_cast<long long>(m) * k;
  const long long kn = static_cast<long long>(k) * n;
  std::vector<double> ha(static_cast<size_t>(mk)), hb(static_cast<size_t>(kn));
  cudaMemcpy(ha.data(), A, static_cast<size_t>(mk) * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), B, static_cast<size_t>(kn) * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (long long i = 0; i < mk; ++i) {
    maxA = std::fmax(maxA, std::fabs(ha[static_cast<size_t>(i)]));
  }
  for (long long i = 0; i < kn; ++i) {
    maxB = std::fmax(maxB, std::fabs(hb[static_cast<size_t>(i)]));
  }
  for (long long i = 0; i < mn; ++i) {
    mx = std::fmax(mx, std::fabs(href[static_cast<size_t>(i)] -
                                  ho[static_cast<size_t>(i)]));
  }
  std::fprintf(stderr, "ozaki2 check %s m=%d n=%d k=%d maxA=%.3e maxB=%.3e max=%.6e\n",
               tag, m, n, k, maxA, maxB, mx);
  cudaFree(ref);
}

static void maybe_check_strided(const char *tag, int m, int n, int k,
                                const double *A, int lda, long long strideA,
                                const double *B, int ldb, long long strideB,
                                const double *C, int ldc, long long strideC,
                                int batch)
{
  if (!std::getenv("OZAKI2_CHECK")) {
    return;
  }
  const long long mn = static_cast<long long>(m) * n;
  const long long total = mn * static_cast<long long>(batch);
  double *ref = nullptr;
  if (check_cuda(cudaMalloc(&ref, static_cast<size_t>(total) * sizeof(double)),
                 "check malloc")) {
    return;
  }
  const int istat = cublas_dgemm_strided_batched_impl(
      0, 0, m, n, k, 1.0, A, lda, strideA, B, ldb, strideB, 0.0, ref, ldc,
      strideC, batch);
  if (istat != 0) {
    std::fprintf(stderr, "ozaki2 check %s: cublas ref failed %d\n", tag, istat);
    cudaFree(ref);
    return;
  }
  std::vector<double> href(static_cast<size_t>(total)),
      ho(static_cast<size_t>(total));
  cudaMemcpy(href.data(), ref, static_cast<size_t>(total) * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(ho.data(), C, static_cast<size_t>(total) * sizeof(double),
             cudaMemcpyDeviceToHost);
  double mx = 0.0;
  for (long long i = 0; i < total; ++i) {
    mx = std::fmax(mx, std::fabs(href[static_cast<size_t>(i)] -
                                  ho[static_cast<size_t>(i)]));
  }
  std::fprintf(stderr, "ozaki2 check %s m=%d n=%d k=%d batch=%d max=%.6e\n", tag,
               m, n, k, batch, mx);
  cudaFree(ref);
}

__global__ void ozaki2_row_max_kernel(const double *__restrict__ mat,
                                      double *__restrict__ scale, int rows,
                                      int cols, int lda)
{
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    double maxv = 0.0;
    for (int c = 0; c < cols; ++c) {
      maxv = fmax(maxv, fabs(mat[row + static_cast<long long>(c) * lda]));
    }
    scale[row] = (maxv > 0.0) ? maxv / 127.0 : 1.0;
  }
}

__global__ void ozaki2_col_max_kernel(const double *__restrict__ mat,
                                      double *__restrict__ scale, int rows,
                                      int cols, int ldb)
{
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
       col += blockDim.x * gridDim.x) {
    double maxv = 0.0;
    for (int r = 0; r < rows; ++r) {
      maxv = fmax(maxv, fabs(mat[r + static_cast<long long>(col) * ldb]));
    }
    scale[col] = (maxv > 0.0) ? maxv / 127.0 : 1.0;
  }
}

__global__ void ozaki2_pack_a_tn_kernel(const double *__restrict__ mat,
                                        signed char *__restrict__ out,
                                        const double *__restrict__ scale, int rows,
                                        int cols, int lda)
{
  const long long total = static_cast<long long>(rows) * cols;
  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int col = static_cast<int>(idx / rows);
    const int row = static_cast<int>(idx % rows);
    const double v = mat[row + static_cast<long long>(col) * lda] / scale[row];
    int q = static_cast<int>(lrint(v));
    q = max(-127, min(127, q));
    out[col + static_cast<long long>(row) * cols] = static_cast<signed char>(q);
  }
}

__global__ void ozaki2_pack_a_batched_tn_kernel(const double *__restrict__ mat,
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
    const double v = slice[row + static_cast<long long>(col) * lda] / sc[row];
    int q = static_cast<int>(lrint(v));
    q = max(-127, min(127, q));
    oslice[col + static_cast<long long>(row) * cols] = static_cast<signed char>(q);
  }
}

__global__ void ozaki2_pack_b_kernel(const double *__restrict__ mat,
                                     signed char *__restrict__ out,
                                     const double *__restrict__ scale, int rows,
                                     int cols, int ldb)
{
  const long long total = static_cast<long long>(rows) * cols;
  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int col = static_cast<int>(idx / rows);
    const int row = static_cast<int>(idx % rows);
    const double v =
        mat[row + static_cast<long long>(col) * ldb] / scale[col];
    int q = static_cast<int>(lrint(v));
    q = max(-127, min(127, q));
    out[row + static_cast<long long>(col) * ldb] = static_cast<signed char>(q);
  }
}

__global__ void ozaki2_row_max_batched_kernel(const double *__restrict__ mat,
                                            double *__restrict__ scale, int rows,
                                            int cols, int lda,
                                            long long stride, int batch)
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
    sc[row] = (maxv > 0.0) ? maxv / 127.0 : 1.0;
  }
}

__global__ void ozaki2_pack_a_batched_kernel(const double *__restrict__ mat,
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
    const double v = slice[row + static_cast<long long>(col) * lda] / sc[row];
    int q = static_cast<int>(lrint(v));
    q = max(-127, min(127, q));
    oslice[col + static_cast<long long>(row) * cols] = static_cast<signed char>(q);
  }
}

static __device__ __host__ int64_t mod_norm(int64_t v, int64_t p)
{
  int64_t r = v % p;
  if (r < 0) r += p;
  return r;
}

__global__ void ozaki2_crt_batched_kernel(double *__restrict__ C,
                                          const int *__restrict__ residues,
                                          const int *__restrict__ moduli,
                                          const int64_t *__restrict__ garner_inv,
                                          const double *__restrict__ scale_a,
                                          const double *__restrict__ scale_b,
                                          int s, int m, int n, int ldc,
                                          long long stride_c, int batch,
                                          int64_t max_int_product, int accumulate)
{
  const long long mn = static_cast<long long>(m) * n;
  const long long residue_stride = mn * static_cast<long long>(batch);
  const long long total = mn * static_cast<long long>(batch);
  for (long long g = blockIdx.x * blockDim.x + threadIdx.x; g < total;
       g += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int b = static_cast<int>(g / mn);
    const long long idx = g - static_cast<long long>(b) * mn;
    const int col = static_cast<int>(idx / m);
    const int row = static_cast<int>(idx % m);
    const long long base = static_cast<long long>(b) * mn;

    int64_t x = static_cast<int64_t>(
        residues[base + idx]);
    int64_t prod = moduli[0];
    for (int i = 1; i < s; ++i) {
      const int64_t pi = moduli[i];
      const int64_t ri =
          static_cast<int64_t>(
              residues[static_cast<long long>(i) * residue_stride + base + idx]);
      int64_t diff = ri - x % pi;
      diff %= pi;
      if (diff < 0) diff += pi;
      const int64_t mult = (diff * garner_inv[i - 1]) % pi;
      x += mult * prod;
      prod *= pi;
      if (prod > max_int_product) {
        if (x > max_int_product) {
          x -= prod;
        }
        break;
      }
    }

    const double sa =
        scale_a[static_cast<long long>(b) * m + row];
    const double val = sa * static_cast<double>(x) * scale_b[col];
    const long long off = row + static_cast<long long>(col) * ldc + b * stride_c;
    if (accumulate) {
      C[off] += val;
    } else {
      C[off] = val;
    }
  }
}

__global__ void ozaki2_residual_a_tn_kernel(const double *__restrict__ mat,
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

__global__ void ozaki2_residual_a_batched_tn_kernel(
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

__global__ void ozaki2_crt_kernel(double *__restrict__ C,
                                  const int *__restrict__ residues,
                                  const int *__restrict__ moduli,
                                  const int64_t *__restrict__ garner_inv,
                                  const double *__restrict__ scale_a,
                                  const double *__restrict__ scale_b,
                                  int s, int m, int n, int ldc,
                                  int64_t max_int_product, int accumulate)
{
  const long long mn = static_cast<long long>(m) * n;
  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < mn;
       idx += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int col = static_cast<int>(idx / m);
    const int row = static_cast<int>(idx % m);

    int64_t x = static_cast<int64_t>(residues[idx]);
    int64_t prod = moduli[0];
    for (int i = 1; i < s; ++i) {
      const int64_t pi = moduli[i];
      const int64_t ri =
          static_cast<int64_t>(
              residues[static_cast<long long>(i) * mn + idx]);
      int64_t diff = ri - x % pi;
      diff %= pi;
      if (diff < 0) diff += pi;
      const int64_t mult = (diff * garner_inv[i - 1]) % pi;
      x += mult * prod;
      prod *= pi;
      if (prod > max_int_product) {
        if (x > max_int_product) {
          x -= prod;
        }
        break;
      }
    }

    const double val =
        scale_a[row] * static_cast<double>(x) * scale_b[col];
    if (accumulate) {
      C[row + static_cast<long long>(col) * ldc] += val;
    } else {
      C[row + static_cast<long long>(col) * ldc] = val;
    }
  }
}

__global__ void ozaki2_split_moduli_kernel(int *__restrict__ residues,
                                           const int *__restrict__ prod,
                                           const int *__restrict__ moduli,
                                           int s, long long n)
{
  for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int64_t val = static_cast<int64_t>(prod[i]);
    for (int j = 0; j < s; ++j) {
      const int64_t p = moduli[j];
      int64_t r = val % p;
      if (r < 0) r += p;
      residues[static_cast<long long>(j) * n + i] = static_cast<int>(r);
    }
  }
}

static int launch_split_moduli(long long n)
{
  const int blocks = static_cast<int>((n + 255) / 256);
  ozaki2_split_moduli_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(
      g_ws.residues, g_ws.prod, g_state.moduli, g_state.moduli_count, n);
  return check_cuda(cudaGetLastError(), "split_moduli");
}

static int launch_row_max(const double *mat, double *scale, int rows, int cols,
                          int lda)
{
  const int blocks = (rows + 255) / 256;
  ozaki2_row_max_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(mat, scale, rows, cols, lda);
  return check_cuda(cudaGetLastError(), "row_max");
}

static int launch_col_max(const double *mat, double *scale, int rows, int cols,
                          int ldb)
{
  const int blocks = (cols + 255) / 256;
  ozaki2_col_max_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(mat, scale, rows, cols, ldb);
  return check_cuda(cudaGetLastError(), "col_max");
}

static int launch_pack_a(const double *mat, signed char *out,
                         const double *scale, int rows, int cols, int lda)
{
  const long long total = static_cast<long long>(rows) * cols;
  const int blocks = static_cast<int>((total + 255) / 256);
  ozaki2_pack_a_tn_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(mat, out, scale, rows, cols, lda);
  return check_cuda(cudaGetLastError(), "pack_a");
}

static int launch_pack_b(const double *mat, signed char *out,
                         const double *scale, int rows, int cols, int ldb)
{
  const long long total = static_cast<long long>(rows) * cols;
  const int blocks = static_cast<int>((total + 255) / 256);
  ozaki2_pack_b_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(mat, out, scale, rows, cols, ldb);
  return check_cuda(cudaGetLastError(), "pack_b");
}

static int launch_row_max_pack_a_batched(const double *mat, int m, int k, int lda,
                                         long long strideA, int batch)
{
  const long long plane = static_cast<long long>(m) * k;
  const long long total_rows =
      static_cast<long long>(m) * static_cast<long long>(batch);
  const int row_blocks = static_cast<int>((total_rows + 255) / 256);
  ozaki2_row_max_batched_kernel<<<row_blocks, 256, 0, dg_cuda_stream>>>(
      mat, g_ws.scale_a, m, k, lda, strideA, batch);
  int err = check_cuda(cudaGetLastError(), "row_max_batched");
  if (err) return err;
  const int pack_blocks = static_cast<int>((plane * static_cast<long long>(batch) + 255) / 256);
  ozaki2_pack_a_batched_tn_kernel<<<pack_blocks, 256, 0, dg_cuda_stream>>>(
      mat, g_ws.iA, g_ws.scale_a, m, k, lda, strideA, batch);
  return check_cuda(cudaGetLastError(), "pack_a_batched");
}

static int launch_residual_a_batched(const double *mat, int m, int k, int lda,
                                     long long strideA, int batch)
{
  const long long plane = static_cast<long long>(m) * k;
  const int pack_blocks = static_cast<int>((plane * static_cast<long long>(batch) + 255) / 256);
  ozaki2_residual_a_batched_tn_kernel<<<pack_blocks, 256, 0, dg_cuda_stream>>>(
      mat, g_ws.iA, g_ws.scale_a, g_ws.res_a, m, k, lda, strideA, batch);
  return check_cuda(cudaGetLastError(), "residual_a_batched");
}

static int launch_crt_batched(double *C, const int *residues, int m, int n,
                              int ldc, long long stride_c, int batch, int k_dim,
                              int accumulate)
{
  const long long mn = static_cast<long long>(m) * n;
  const long long total = mn * static_cast<long long>(batch);
  const int blocks = static_cast<int>((total + 255) / 256);
  const int64_t max_int =
      static_cast<int64_t>(k_dim) * 127LL * 127LL;
  ozaki2_crt_batched_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(
      C, residues, g_state.moduli, g_state.garner_inv, g_ws.scale_a,
      g_ws.scale_b, g_state.moduli_count, m, n, ldc, stride_c, batch, max_int,
      accumulate);
  return check_cuda(cudaGetLastError(), "crt_batched");
}

static int launch_crt(double *C, const int *residues, int m, int n, int ldc,
                      int k_dim, int accumulate)
{
  const long long mn = static_cast<long long>(m) * n;
  const int blocks = static_cast<int>((mn + 255) / 256);
  const int64_t max_int =
      static_cast<int64_t>(k_dim) * 127LL * 127LL;
  ozaki2_crt_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(
      C, residues, g_state.moduli, g_state.garner_inv, g_ws.scale_a,
      g_ws.scale_b, g_state.moduli_count, m, n, ldc, max_int, accumulate);
  return check_cuda(cudaGetLastError(), "crt");
}

extern "C" int ozaki2_init(int moduli_count, int fixed_mantissa)
{
  if (moduli_count < kMinModuli || moduli_count > kMaxModuli) {
    std::fprintf(stderr,
                 "ozaki2_init: moduli_count must be in [%d, %d], got %d\n",
                 kMinModuli, kMaxModuli, moduli_count);
    return -1;
  }
  g_state.moduli_count = moduli_count;
  g_state.fixed = fixed_mantissa ? 1 : 0;
  for (int i = 0; i < moduli_count; ++i) {
    g_state.moduli[i] = kModuliPool[i];
  }
  precompute_garner_inverses(g_state);
  g_state.inited = 1;
  unsigned __int128 prod = 1;
  for (int i = 0; i < moduli_count; ++i) {
    prod *= static_cast<unsigned __int128>(g_state.moduli[i]);
  }
  const double bits = std::log2(static_cast<double>(prod));
  if (g_state.fixed) {
    std::printf(
        "Ozaki-II: FIXED %d moduli (CRT product %.1f bits, single A pack)\n",
        moduli_count, bits);
  } else {
    std::printf(
        "Ozaki-II: DYNAMIC %d moduli (CRT product %.1f bits, A residual up to %d)\n",
        moduli_count, bits, kMaxASlices);
  }
  return 0;
}

extern "C" int ozaki2_finalize(void)
{
  ozaki2_free_workspace();
  g_state = Ozaki2State{};
  return 0;
}

extern "C" int ozaki2_alloc_workspace(int Nq, int Ne, int Np)
{
  if (!g_state.inited) {
  const int rc = ozaki2_init(kDefaultModuli, 1);
    if (rc != 0) return rc;
  }

  ozaki2_free_workspace();

  g_ws.Nq = Nq;
  g_ws.Ne = Ne;
  g_ws.Np = Np;
  g_ws.nq2 = Nq * Nq;
  g_ws.mn = static_cast<long long>(Np) * Ne;
  g_ws.k_max = Nq;
  g_ws.max_batch = std::max(Nq * Ne, Ne);

  const long long k_rows = static_cast<long long>(Nq) * Nq;
  const long long mn = g_ws.mn;
  const int s = g_state.moduli_count;
  const int max_batch = g_ws.max_batch;

  g_ws.iA_bytes = static_cast<size_t>(k_rows * max_batch);
  g_ws.iB_bytes = static_cast<size_t>(mn);

  const size_t scale_a_bytes =
      static_cast<size_t>(Nq) * static_cast<size_t>(max_batch) * sizeof(double);
  const size_t scale_b_bytes = static_cast<size_t>(mn) * sizeof(double);
  const size_t residue_bytes =
      static_cast<size_t>(s) * static_cast<size_t>(mn) * sizeof(int);
  const size_t prod_bytes = static_cast<size_t>(mn) * sizeof(int);
  g_ws.res_a_bytes = static_cast<size_t>(mn) * sizeof(double);

  int err = 0;
  err = check_cuda(cudaMalloc(&g_ws.iA, g_ws.iA_bytes), "malloc iA");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.iB, g_ws.iB_bytes), "malloc iB");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.scale_a, scale_a_bytes), "malloc scale_a");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.scale_b, scale_b_bytes), "malloc scale_b");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.residues, residue_bytes), "malloc residues");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.prod, prod_bytes), "malloc prod");
  if (err) return err;
  err = check_cuda(cudaMalloc(&g_ws.res_a, g_ws.res_a_bytes), "malloc res_a");
  if (err) return err;

  std::printf("ozaki2 workspace: Nq=%d Ne=%d Np=%d s=%d residues=%.1f MB\n",
              Nq, Ne, Np, s, residue_bytes / 1.0e6);

  return 0;
}

extern "C" void ozaki2_free_workspace(void)
{
  if (g_ws.iA) cudaFree(g_ws.iA);
  if (g_ws.iB) cudaFree(g_ws.iB);
  if (g_ws.scale_a) cudaFree(g_ws.scale_a);
  if (g_ws.scale_b) cudaFree(g_ws.scale_b);
  if (g_ws.residues) cudaFree(g_ws.residues);
  if (g_ws.prod) cudaFree(g_ws.prod);
  if (g_ws.res_a) cudaFree(g_ws.res_a);
  g_ws = Ozaki2Workspace{};
}

static int ensure_workspace(int m, int n, int k)
{
  if (!g_ws.residues) {
    std::fprintf(stderr, "ozaki2: workspace not allocated\n");
    return -1;
  }
  const long long mn = static_cast<long long>(m) * n;
  if (mn > g_ws.mn) {
    std::fprintf(stderr, "ozaki2: GEMM mn=%lld exceeds workspace mn=%lld\n",
                 static_cast<long long>(mn), g_ws.mn);
    return -1;
  }
  if (static_cast<long long>(m) > g_ws.k_max &&
      static_cast<long long>(n) > g_ws.k_max &&
      static_cast<long long>(k) > g_ws.k_max) {
    std::fprintf(stderr, "ozaki2: dimension exceeds workspace\n");
    return -1;
  }
  g_ws.k_max = std::max(g_ws.k_max, static_cast<long long>(k));
  return 0;
}

extern "C" int ozaki2_dgemm(int transa, int transb, int m, int n, int k,
                            const double *A, int lda, const double *B, int ldb,
                            double *C, int ldc)
{
  if (!g_state.inited) {
    return -1;
  }
  if (transa != 0 || transb != 0) {
    std::fprintf(stderr, "ozaki2_dgemm: only NN supported\n");
    return -1;
  }
  if (ensure_workspace(m, n, k) != 0) return -1;

  int err = launch_col_max(B, g_ws.scale_b, k, n, ldb);
  if (err) return err;
  err = launch_pack_b(B, g_ws.iB, g_ws.scale_b, k, n, ldb);
  if (err) return err;

  const double *cur_a = A;
  const int max_a_slices = g_state.fixed ? 1 : kMaxASlices;
  for (int slice = 0; slice < max_a_slices; ++slice) {
    if (slice > 0) {
      cur_a = g_ws.res_a;
    }
    err = launch_row_max(cur_a, g_ws.scale_a, m, k, lda);
    if (err) return err;
    err = launch_pack_a(cur_a, g_ws.iA, g_ws.scale_a, m, k, lda);
    if (err) return err;
    err = run_int8_gemm_crt(m, n, k, C, ldc, slice > 0);
    if (err) return err;
    if (g_state.fixed || !matrix_needs_second_slice(m, 1) ||
        slice == max_a_slices - 1) {
      break;
    }
    const long long mk = static_cast<long long>(m) * k;
    const int blocks = static_cast<int>((mk + 255) / 256);
  if (slice == 0) {
      ozaki2_residual_a_tn_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(
          A, g_ws.iA, g_ws.scale_a, g_ws.res_a, m, k, lda);
    } else {
      ozaki2_residual_a_tn_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(
          g_ws.res_a, g_ws.iA, g_ws.scale_a, g_ws.res_a, m, k, lda);
    }
    err = check_cuda(cudaGetLastError(), "residual_a");
    if (err) return err;
  }

  err = sync_ozaki2_stream();
  if (err) return err;
  maybe_check_dgemm("dgemm", m, n, k, A, lda, B, ldb, C, ldc);
  return 0;
}

extern "C" int ozaki2_dgemm_strided_batched(int transa, int transb, int m, int n,
                                            int k, const double *A, int lda,
                                            long long strideA, const double *B,
                                            int ldb, long long strideB,
                                            double *C, int ldc,
                                            long long strideC, int batch)
{
  if (!g_state.inited) {
    return -1;
  }
  if (transa != 0 || transb != 0) {
    std::fprintf(stderr, "ozaki2_dgemm_strided_batched: only NN supported\n");
    return -1;
  }
  if (ensure_workspace(m, n, k) != 0) return -1;
  if (static_cast<long long>(m) * n * static_cast<long long>(batch) > g_ws.mn) {
    std::fprintf(stderr, "ozaki2: batched output exceeds workspace\n");
    return -1;
  }
  if (batch > g_ws.max_batch) {
    std::fprintf(stderr, "ozaki2: batch=%d exceeds workspace max_batch=%d\n",
                 batch, g_ws.max_batch);
    return -1;
  }

  int err = launch_col_max(B, g_ws.scale_b, k, n, ldb);
  if (err) return err;
  err = launch_pack_b(B, g_ws.iB, g_ws.scale_b, k, n, ldb);
  if (err) return err;

  const double *cur_a = A;
  const int max_a_slices = g_state.fixed ? 1 : kMaxASlices;
  for (int slice = 0; slice < max_a_slices; ++slice) {
    if (slice > 0) {
      cur_a = g_ws.res_a;
    }
    if (strideA > 0) {
      err = launch_row_max_pack_a_batched(cur_a, m, k, lda, strideA, batch);
    } else {
      err = launch_row_max(cur_a, g_ws.scale_a, m, k, lda);
      if (err) return err;
      err = launch_pack_a(cur_a, g_ws.iA, g_ws.scale_a, m, k, lda);
    }
    if (err) return err;

    err = run_int8_strided_gemm_crt(m, n, k, C, ldc, strideA, strideB, strideC,
                                    batch, slice > 0);
    if (err) return err;

    if (g_state.fixed || !matrix_needs_second_slice(m, batch) ||
        slice == max_a_slices - 1) {
      break;
    }
    if (!g_ws.res_a) {
      break;
    }

    const double *res_src = (slice == 0) ? A : g_ws.res_a;
    if (strideA > 0) {
      err = launch_residual_a_batched(res_src, m, k, lda, strideA, batch);
    } else {
      const long long mk = static_cast<long long>(m) * k;
      const int blocks = static_cast<int>((mk + 255) / 256);
      ozaki2_residual_a_tn_kernel<<<blocks, 256, 0, dg_cuda_stream>>>(
          res_src, g_ws.iA, g_ws.scale_a, g_ws.res_a, m, k, lda);
      err = check_cuda(cudaGetLastError(), "residual_a");
    }
    if (err) return err;
  }

  err = sync_ozaki2_stream();
  if (err) return err;
  maybe_check_strided("strided", m, n, k, A, lda, strideA, B, ldb, strideB, C,
                      ldc, strideC, batch);
  return 0;
}
