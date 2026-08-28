// Why is cuBLAS FP64 fixed-point emulation still slower than native DGEMM
// at the p=1023 volume-GEMM shapes?
//
// Isolates the three DG calls (x / y-batched / z) plus a large square that
// NVIDIA's heatmap claims as a win, without the 176 GiB DG field payload.
//
// Build (login node is fine; no nsys/ncu):
//   module load nvhpc
//   nvcc -O3 -arch=sm_100 -std=c++17 -o p1023_emu_why p1023_emu_why.cu -lcublas

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x)                                                              \
  do {                                                                     \
    cudaError_t e = (x);                                                   \
    if (e != cudaSuccess) {                                                \
      std::printf("cuda %s @%d: %s\n", #x, __LINE__,                       \
                  cudaGetErrorString(e));                                  \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)
#define BK(x)                                                              \
  do {                                                                     \
    cublasStatus_t s = (x);                                                \
    if (s != CUBLAS_STATUS_SUCCESS) {                                      \
      std::printf("cublas %s @%d status=%d\n", #x, __LINE__, (int)s);      \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

static void mem_info(const char *tag)
{
  size_t free_b = 0, tot_b = 0;
  CK(cudaMemGetInfo(&free_b, &tot_b));
  std::printf("# mem %-16s  free=%.2f GiB  total=%.2f GiB\n", tag,
              free_b / (1024.0 * 1024.0 * 1024.0),
              tot_b / (1024.0 * 1024.0 * 1024.0));
}

template <class F>
static double med_us(F body, int iters, int warm)
{
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  for (int i = 0; i < warm; ++i)
    body();
  CK(cudaDeviceSynchronize());
  std::vector<double> us;
  us.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    float ms = 0.f;
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

static void fill_chebyshev_d(double *h, int nq)
{
  const int n = nq - 1;
  std::vector<double> x(nq), c(nq);
  for (int i = 0; i < nq; ++i)
    x[i] = std::cos(M_PI * i / (double)n);
  c[0] = 2.0;
  c[n] = 2.0;
  for (int i = 1; i < n; ++i)
    c[i] = 1.0;
  for (int i = 0; i < nq; ++i) {
    for (int j = 0; j < nq; ++j) {
      if (i == j) {
        if (i == 0)
          h[i + nq * j] = (2.0 * n * n + 1.0) / 6.0;
        else if (i == n)
          h[i + nq * j] = -(2.0 * n * n + 1.0) / 6.0;
        else
          h[i + nq * j] = -x[i] / (2.0 * (1.0 - x[i] * x[i]));
      } else {
        const double cij = c[i] / c[j];
        const double sgn = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        h[i + nq * j] = cij * sgn / (x[i] - x[j]);
      }
    }
  }
}

static void report_range(const char *tag, const double *h, int n)
{
  double amax = 0.0, amin = 1e300;
  int nz = 0;
  for (int i = 0; i < n; ++i) {
    const double v = std::fabs(h[i]);
    if (v == 0.0)
      continue;
    ++nz;
    amax = std::max(amax, v);
    amin = std::min(amin, v);
  }
  const double bits = (amax > 0.0 && amin > 0.0) ? std::log2(amax / amin) : 0.0;
  std::printf("# range %-12s  |min|=%.3e  |max|=%.3e  log2(max/min)=%.1f  nz=%d/%d\n",
              tag, amin, amax, bits, nz, n);
}

#if CUBLAS_VERSION >= 130002
static void dump_mode(cublasHandle_t h, const char *tag)
{
  cublasMath_t math = CUBLAS_DEFAULT_MATH;
  cublasEmulationStrategy_t strat = CUBLAS_EMULATION_STRATEGY_DEFAULT;
  BK(cublasGetMathMode(h, &math));
  BK(cublasGetEmulationStrategy(h, &strat));
  std::printf("# mode %-40s  math=%d  strategy=%d\n", tag, (int)math,
              (int)strat);
}

static void configure_emu(cublasHandle_t h, int eager, int mantissa_bits,
                          int *bits_out)
{
  BK(cublasSetMathMode(h, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
  BK(cublasSetEmulationStrategy(
      h, eager ? CUBLAS_EMULATION_STRATEGY_EAGER
               : CUBLAS_EMULATION_STRATEGY_DEFAULT));
  if (mantissa_bits > 0) {
    BK(cublasSetFixedPointEmulationMantissaControl(
        h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
    BK(cublasSetFixedPointEmulationMaxMantissaBitCount(h, mantissa_bits));
  } else {
    BK(cublasSetFixedPointEmulationMantissaControl(
        h, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
  }
  if (bits_out) {
    *bits_out = -1;
    BK(cublasSetFixedPointEmulationMantissaBitCountPointer(h, bits_out));
  }
}

static void configure_native(cublasHandle_t h)
{
  BK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
  BK(cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_DEFAULT));
  BK(cublasSetFixedPointEmulationMantissaBitCountPointer(h, nullptr));
}
#else
#error "cuBLAS 13.0u2+ required"
#endif

static constexpr int kNq = 1024;
static constexpr int kNq2 = kNq * kNq; // 1,048,576
static constexpr int kWarm = 2;
static constexpr int kIters = 5;

struct Buf {
  double *A = nullptr; // Nq * Nq
  double *B = nullptr; // Nq * Nq2   (flux / wide B)
  double *C = nullptr; // Nq * Nq2
  signed char *iA = nullptr;
  signed char *iB = nullptr;
  int *iC = nullptr;
  void *workspace = nullptr;
};

static void alloc_all(Buf &b, int with_ws, int with_int8)
{
  CK(cudaMalloc(&b.A, (size_t)kNq * kNq * sizeof(double)));
  CK(cudaMalloc(&b.B, (size_t)kNq * (size_t)kNq2 * sizeof(double)));
  CK(cudaMalloc(&b.C, (size_t)kNq * (size_t)kNq2 * sizeof(double)));
  mem_info("after ABC");
  if (with_ws) {
    CK(cudaMalloc(&b.workspace, size_t{8} << 30));
    mem_info("after 8GiB ws");
  }
  if (with_int8) {
    CK(cudaMalloc(&b.iA, (size_t)kNq * kNq));
    CK(cudaMalloc(&b.iB, (size_t)kNq * (size_t)kNq2));
    CK(cudaMalloc(&b.iC, (size_t)kNq * (size_t)kNq2 * sizeof(int)));
    CK(cudaMemset(b.iA, 1, (size_t)kNq * kNq));
    CK(cudaMemset(b.iB, 1, (size_t)kNq * (size_t)kNq2));
    mem_info("after int8");
  }
}

static void time_print(const char *name, double us, double flops)
{
  const double tflops = flops / (us * 1e-6) / 1e12;
  std::printf("  %-28s %9.1f us  %6.2f TFLOP/s\n", name, us, tflops);
}

int main(int argc, char **argv)
{
  int want_ws = 1;
  int want_int8 = 1;
  int want_square = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-ws") == 0)
      want_ws = 0;
    if (std::strcmp(argv[i], "--no-int8") == 0)
      want_int8 = 0;
    if (std::strcmp(argv[i], "--no-square") == 0)
      want_square = 0;
  }

  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  int best = 0;
  size_t best_free = 0;
  for (int d = 0; d < ndev; ++d) {
    CK(cudaSetDevice(d));
    size_t free_b = 0, tot_b = 0;
    CK(cudaMemGetInfo(&free_b, &tot_b));
    std::printf("# gpu %d  free=%.2f GiB  total=%.2f GiB\n", d,
                free_b / (1024.0 * 1024.0 * 1024.0),
                tot_b / (1024.0 * 1024.0 * 1024.0));
    if (free_b > best_free) {
      best_free = free_b;
      best = d;
    }
  }
  CK(cudaSetDevice(best));
  std::printf("# using gpu %d\n", best);
  mem_info("start");

  Buf b;
  alloc_all(b, want_ws, want_int8);

  std::vector<double> hD((size_t)kNq * kNq);
  fill_chebyshev_d(hD.data(), kNq);
  report_range("chebyshevD", hD.data(), kNq * kNq);
  CK(cudaMemcpy(b.A, hD.data(), hD.size() * sizeof(double),
                cudaMemcpyHostToDevice));
  // flux-like B: smooth field, not full mantissa soup
  CK(cudaMemset(b.B, 0x3f, (size_t)kNq * (size_t)kNq2 * sizeof(double)));
  CK(cudaMemset(b.C, 0, (size_t)kNq * (size_t)kNq2 * sizeof(double)));

  const int emu_from_env =
      (std::getenv("CUBLAS_EMULATE_DOUBLE_PRECISION") &&
       std::getenv("CUBLAS_EMULATE_DOUBLE_PRECISION")[0] == '1');
  if (!emu_from_env) {
    setenv("CUBLAS_EMULATE_DOUBLE_PRECISION", "0", 1);
    setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "0", 1);
  } else {
    std::printf("# CUBLAS_EMULATE_DOUBLE_PRECISION already 1 at process start\n");
  }
  cublasHandle_t h;
  BK(cublasCreate(&h));
  if (b.workspace)
    BK(cublasSetWorkspace(h, b.workspace, size_t{8} << 30));

  const double d1 = 1.0, d0 = 0.0;
  const int i1 = 1, i0 = 0;
  const double flop_x = 2.0 * kNq * (double)kNq2 * kNq; // 2 Nq^4
  const double flop_y = flop_x;                         // batch Nq of Nq^3
  const double flop_z = flop_x;
  const long long stride_plane = (long long)kNq * kNq;
  const long long stride_zero = 0;

  auto gemm_x = [&] {
    BK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, kNq, kNq2, kNq, &d1, b.A, kNq,
                   b.B, kNq, &d0, b.C, kNq));
  };
  auto gemm_y = [&] {
    BK(cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, kNq, kNq, kNq,
                                 &d1, b.B, kNq, stride_plane, b.A, kNq,
                                 stride_zero, &d0, b.C, kNq, stride_plane,
                                 kNq));
  };
  auto gemm_z = [&] {
    BK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, kNq2, kNq, kNq, &d1, b.B, kNq2,
                   b.A, kNq, &d0, b.C, kNq2));
  };

  auto banner = [](const char *s) {
    std::printf("\n== %s ==\n", s);
  };

  int bits_used = -1;

  if (!emu_from_env) {
    banner("native FP64 (DEFAULT math, no eager)");
    configure_native(h);
    dump_mode(h, "native");
    time_print("x 1024x1048576x1024", med_us(gemm_x, kIters, kWarm), flop_x);
    time_print("y batched 1024^3 x1024", med_us(gemm_y, kIters, kWarm), flop_y);
    time_print("z 1048576x1024x1024", med_us(gemm_z, kIters, kWarm), flop_z);

    BK(cublasDestroy(h));
    setenv("CUBLAS_EMULATION_STRATEGY", "eager", 1);
    setenv("CUBLAS_EMULATE_DOUBLE_PRECISION", "1", 1);
    BK(cublasCreate(&h));
    if (b.workspace)
      BK(cublasSetWorkspace(h, b.workspace, size_t{8} << 30));
  }

  banner("EAGER + DYNAMIC mantissa (repo CublasEmulation=.true.)");
  configure_emu(h, /*eager=*/1, /*bits=*/0, &bits_used);
  dump_mode(h, "eager+dynamic");
  time_print("x", med_us(gemm_x, kIters, kWarm), flop_x);
  std::printf("    bits_used after x = %d\n", bits_used);
  time_print("y", med_us(gemm_y, kIters, kWarm), flop_y);
  std::printf("    bits_used after y = %d\n", bits_used);
  time_print("z", med_us(gemm_z, kIters, kWarm), flop_z);
  std::printf("    bits_used after z = %d\n", bits_used);

  banner("EAGER + FIXED 23/55 on y and z (isolate ADP tax vs slice count)");
  for (int bits : {23, 55}) {
    configure_emu(h, 1, bits, &bits_used);
    char lx[64], ly[64], lz[64];
    std::snprintf(lx, sizeof(lx), "x fixed %d", bits);
    std::snprintf(ly, sizeof(ly), "y fixed %d", bits);
    std::snprintf(lz, sizeof(lz), "z fixed %d", bits);
    time_print(lx, med_us(gemm_x, kIters, kWarm), flop_x);
    time_print(ly, med_us(gemm_y, kIters, kWarm), flop_y);
    time_print(lz, med_us(gemm_z, kIters, kWarm), flop_z);
    std::printf("    bits_used = %d\n", bits_used);
  }

  banner("DEFAULT strategy + DYNAMIC (PERFORMANT / ADP may pick native)");
  configure_emu(h, /*eager=*/0, /*bits=*/0, &bits_used);
  dump_mode(h, "default+dynamic");
  time_print("x", med_us(gemm_x, kIters, kWarm), flop_x);
  std::printf("    bits_used after x = %d\n", bits_used);
  time_print("y", med_us(gemm_y, kIters, kWarm), flop_y);
  std::printf("    bits_used after y = %d\n", bits_used);
  time_print("z", med_us(gemm_z, kIters, kWarm), flop_z);
  std::printf("    bits_used after z = %d\n", bits_used);

  banner("EAGER + FIXED mantissa sweep (x GEMM only)");
  for (int bits : {23, 39, 47, 55, 63, 79}) {
    configure_emu(h, 1, bits, &bits_used);
    const double us = med_us(gemm_x, kIters, kWarm);
    char label[64];
    std::snprintf(label, sizeof(label), "x fixed %d bits", bits);
    time_print(label, us, flop_x);
    std::printf("    bits_used = %d\n", bits_used);
  }

  banner("control: same K=1024, shrink N (does Dgemm actually emulate?)");
  {
    const int ncap = 65536;
    const double flop_c = 2.0 * kNq * (double)ncap * kNq;
    auto gemm_cap = [&] {
      BK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, kNq, ncap, kNq, &d1, b.A, kNq,
                     b.B, kNq, &d0, b.C, kNq));
    };
    auto gemm_cap_ex = [&] {
      BK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kNq, ncap, kNq, &d1, b.A,
                      CUDA_R_64F, kNq, b.B, CUDA_R_64F, kNq, &d0, b.C,
                      CUDA_R_64F, kNq, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
                      CUBLAS_GEMM_DEFAULT));
    };
    configure_native(h);
    time_print("Dgemm native N=64k", med_us(gemm_cap, kIters, kWarm), flop_c);
    configure_emu(h, 1, 55, &bits_used);
    dump_mode(h, "eager+fixed55 for N=64k");
    time_print("Dgemm EAGER55 N=64k", med_us(gemm_cap, kIters, kWarm), flop_c);
    std::printf("    bits_used = %d\n", bits_used);
    time_print("GemmEx FORCE N=64k", med_us(gemm_cap_ex, kIters, kWarm), flop_c);
    std::printf("    bits_used = %d\n", bits_used);
  }

  banner("force GemmEx EMULATED_FIXEDPOINT on full x shape");
  {
    auto gemm_x_ex = [&] {
      BK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kNq, kNq2, kNq, &d1, b.A,
                      CUDA_R_64F, kNq, b.B, CUDA_R_64F, kNq, &d0, b.C,
                      CUDA_R_64F, kNq, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
                      CUBLAS_GEMM_DEFAULT));
    };
    configure_emu(h, 1, 55, &bits_used);
    time_print("GemmEx FORCE full x", med_us(gemm_x_ex, 3, 1), flop_x);
    std::printf("    bits_used = %d\n", bits_used);
  }

  banner("N sweep Dgemm EAGER-55 vs native (K=M=1024)");
  configure_native(h);
  for (int n : {4096, 16384, 65536, 262144, 1048576}) {
    const double flop_n = 2.0 * kNq * (double)n * kNq;
    auto g = [&] {
      BK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, kNq, n, kNq, &d1, b.A, kNq,
                     b.B, kNq, &d0, b.C, kNq));
    };
    configure_native(h);
    const double t_nat = med_us(g, 3, 1);
    configure_emu(h, 1, 55, &bits_used);
    const double t_emu = med_us(g, 3, 1);
    std::printf("  N=%7d  native %8.1f us  eager55 %8.1f us  ratio %.3f  bits=%d\n",
                n, t_nat, t_emu, t_emu / t_nat, bits_used);
  }

  if (want_int8 && b.iC) {
    banner("INT8 arithmetic floor on x shape (no pack, no reconstruct)");
    auto int8_once = [&] {
      BK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kNq, kNq2, kNq, &i1, b.iA,
                      CUDA_R_8I, kNq, b.iB, CUDA_R_8I, kNq, &i0, b.iC,
                      CUDA_R_32I, kNq, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    };
    configure_native(h);
    const double us1 = med_us(int8_once, kIters, kWarm);
    time_print("int8 x1", us1, flop_x);
    const double top = flop_x / (us1 * 1e-6) / 1e12;
    std::printf("    INT8/FP64 rate: compare to native x TFLOP/s above\n");
    std::printf("    s=7  arithmetic floor ~ %.1f us  (7 * int8)\n", 7.0 * us1);
    std::printf("    s=10 arithmetic floor ~ %.1f us  (10 * int8)\n", 10.0 * us1);
    std::printf("    Scheme-I s=7 square (49 GEMMs) floor ~ %.1f us\n",
                49.0 * us1);
    (void)top;
  }

  if (want_square) {
    // 8192^3 fits in the existing B/C buffers? 8192^2 * 8 = 512 MiB, yes in A
    // if we repurpose B as A/B/C slices. Use the beginning of B/C.
    const int ns = 8192;
    if ((size_t)ns * ns * sizeof(double) * 3 <=
        (size_t)kNq * (size_t)kNq2 * sizeof(double)) {
      banner("square 8192^3 (NVIDIA heatmap 'large' regime)");
      double *As = b.B;
      double *Bs = b.B + (size_t)ns * ns;
      double *Cs = b.C;
      const double flop_s = 2.0 * ns * (double)ns * ns;
      auto gemm_s = [&] {
        BK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, ns, ns, ns, &d1, As, ns, Bs,
                       ns, &d0, Cs, ns));
      };
      configure_native(h);
      time_print("native square", med_us(gemm_s, kIters, kWarm), flop_s);
      configure_emu(h, 1, 0, &bits_used);
      time_print("EAGER dynamic", med_us(gemm_s, kIters, kWarm), flop_s);
      std::printf("    bits_used = %d\n", bits_used);
      configure_emu(h, 1, 55, &bits_used);
      time_print("EAGER fixed 55", med_us(gemm_s, kIters, kWarm), flop_s);
      std::printf("    bits_used = %d\n", bits_used);
      configure_emu(h, 0, 0, &bits_used);
      time_print("DEFAULT dynamic", med_us(gemm_s, kIters, kWarm), flop_s);
      std::printf("    bits_used = %d\n", bits_used);
    }
  }

  std::printf("\n# done\n");
  return 0;
}
