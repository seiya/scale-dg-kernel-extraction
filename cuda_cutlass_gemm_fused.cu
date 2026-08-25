#include <cuda_runtime.h>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/device_kernel.h"
#include "cutlass/arch/synclog.hpp"

#include "cutlass_z_gemm_assembly.h"

// Volume GEMMs match the cuBLAS nsys kernels:
//   cutlass_80_tensorop_d884gemm_64x128_16x3_nn_align1
//   cutlass_80_tensorop_d884gemm_64x64_16x4_nn_align1
//   cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1

//- Defined in cuda_dg_kernels_tc.cu; the stream shared by the whole CUDA path.
extern cudaStream_t dg_cuda_stream;

namespace {

using ColumnMajor = cutlass::layout::ColumnMajor;
using TensorOp = cutlass::arch::OpClassTensorOp;
using Sm80 = cutlass::arch::Sm80;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using BatchedSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<double, 1, double, double>;

using GemmNN_64x128_3 = cutlass::gemm::device::Gemm<
    double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
    TensorOp, Sm80, cutlass::gemm::GemmShape<64, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>, cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueOp, Swizzle, 3>;

using GemmBatchedNN_64x64_4 = cutlass::gemm::device::GemmBatched<
    double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
    TensorOp, Sm80, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueOp, BatchedSwizzle, 4>;

using GemmBatchedNN_64x32_4 = cutlass::gemm::device::GemmBatched<
    double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
    TensorOp, Sm80, cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<8, 8, 4>,
    EpilogueOp, BatchedSwizzle, 4>;

int cutlass_error(const char *what, cutlass::Status st)
{
  if (st == cutlass::Status::kSuccess) {
    return 0;
  }
  std::fprintf(stderr, "%s: cutlass status %d\n", what, static_cast<int>(st));
  return static_cast<int>(st);
}

template <class Gemm>
int run_gemm_nn(int m, int n, int k, double const *A, int lda, double const *B,
                int ldb, double *C, int ldc)
{
  Gemm gemm_op;
  typename Gemm::Arguments args({m, n, k}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc},
                                {1.0, 0.0});
  const cutlass::Status can = gemm_op.can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("can_implement", can);
  }
  return cutlass_error("gemm", gemm_op(args, nullptr, dg_cuda_stream));
}

template <class GemmBatched>
int run_gemm_batched_nn(int m, int n, int k, double const *A, int lda,
                        long long strideA, double const *B, int ldb,
                        long long strideB, double *C, int ldc, long long strideC,
                        int batch)
{
  GemmBatched gemm_op;
  typename GemmBatched::Arguments args({m, n, k}, {A, lda}, strideA, {B, ldb},
                                       strideB, {C, ldc}, strideC, {C, ldc},
                                       strideC, {1.0, 0.0}, batch);
  const cutlass::Status can = gemm_op.can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("batched can_implement", can);
  }
  return cutlass_error("batched gemm", gemm_op(args, nullptr, dg_cuda_stream));
}

int run_volume_gemms_xy(double *deriv_x, double *deriv_y, const double *flux_x,
                        const double *flux_y, const double *D1D,
                        const double *D1D_tr, int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const long long stride_plane = nq2;

  int st = run_gemm_nn<GemmNN_64x128_3>(Nq, nq2 * Ne, Nq, D1D, Nq, flux_x, Nq,
                                        deriv_x, Nq);
  if (st != 0) {
    return st;
  }

  return run_gemm_batched_nn<GemmBatchedNN_64x64_4>(
      Nq, Nq, Nq, flux_y, Nq, stride_plane, D1D_tr, Nq, 0, deriv_y, Nq,
      stride_plane, Nq * Ne);
}

int run_volume_gemms(double *deriv_x, double *deriv_y, double *deriv_z,
                     const double *flux_x, const double *flux_y,
                     const double *flux_z, const double *D1D,
                     const double *D1D_tr, int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const int Np = nq2 * Nq;
  const long long stride_vol = Np;

  int st = run_volume_gemms_xy(deriv_x, deriv_y, flux_x, flux_y, D1D, D1D_tr, Nq,
                               Ne);
  if (st != 0) {
    return st;
  }

  return run_gemm_batched_nn<GemmBatchedNN_64x32_4>(
      nq2, Nq, Nq, flux_z, nq2, stride_vol, D1D_tr, Nq, 0, deriv_z, nq2,
      stride_vol, Ne);
}

int run_z_gemm_assembly(double *dqdt, const double *flux_z, const double *D1D_tr,
                        const double *deriv_x, const double *deriv_y,
                        const double *lift, const double *escale, int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const int Np = nq2 * Nq;
  const int npoint = Np * Ne;
  const long long stride_vol = Np;
  const int m = nq2;
  const int n = Nq;
  const int k = Nq;

  using GemmZ = GemmBatchedNN_64x32_4;
  using Kernel = GemmBatchedDqdtAssembly<typename GemmZ::GemmKernel::Mma,
                                         typename GemmZ::GemmKernel::Epilogue,
                                         BatchedSwizzle>;

  cutlass::TensorRef<double const, ColumnMajor> ref_A(flux_z, m);
  cutlass::TensorRef<double const, ColumnMajor> ref_B(D1D_tr, k);
  cutlass::TensorRef<double, ColumnMajor> ref_D(dqdt, m);

  typename GemmZ::Arguments gemm_args({m, n, k}, ref_A, stride_vol, ref_B, 0,
                                      ref_D, stride_vol, ref_D, stride_vol,
                                      {1.0, 0.0}, Ne);
  cutlass::Status can = GemmZ::can_implement(gemm_args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("z-assembly can_implement", can);
  }

  auto uargs = GemmZ::to_underlying_arguments(gemm_args);
  BatchedSwizzle swizzle;
  cutlass::gemm::GemmCoord grid_shape = swizzle.get_tiled_shape(
      uargs.problem_size,
      {GemmZ::ThreadblockShape::kM, GemmZ::ThreadblockShape::kN,
       GemmZ::ThreadblockShape::kK},
      uargs.batch_count);

  typename Kernel::Params params;
  params.gemm = typename Kernel::BaseKernel::Params(
      uargs.problem_size, grid_shape, uargs.ref_A.non_const_ref(), uargs.stride_A,
      uargs.ref_B.non_const_ref(), uargs.stride_B, uargs.ref_C.non_const_ref(),
      uargs.stride_C, uargs.ref_D, uargs.stride_D, uargs.epilogue,
      uargs.batch_count);
  params.ptr_dx = deriv_x;
  params.ptr_dy = deriv_y;
  params.ptr_lift = lift;
  params.ptr_ex = escale;
  params.ptr_ey = escale + npoint;
  params.ptr_ez = escale + 2 * npoint;

  dim3 grid = swizzle.get_grid_shape(grid_shape);
  dim3 block(Kernel::kThreadCount, 1, 1);
  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr != cudaSuccess) {
      return 1;
    }
  }

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel><<<grid, block, smem_size, dg_cuda_stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : 1;
}

} // namespace

extern "C" int launch_volume_gemm_cute(
    double *deriv_x, double *deriv_y, double *deriv_z, const double *flux_x,
    const double *flux_y, const double *flux_z, const double *D1D,
    const double *D1D_tr, int Nq, int Ne)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  return run_volume_gemms(deriv_x, deriv_y, deriv_z, flux_x, flux_y, flux_z, D1D,
                          D1D_tr, Nq, Ne);
}

extern "C" int launch_volume_gemm_xy(
    double *deriv_x, double *deriv_y, const double *flux_x, const double *flux_y,
    const double *D1D, const double *D1D_tr, int Nq, int Ne)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  return run_volume_gemms_xy(deriv_x, deriv_y, flux_x, flux_y, D1D, D1D_tr, Nq,
                             Ne);
}

extern "C" int launch_z_gemm_assembly(
    double *dqdt, const double *flux_z, const double *D1D_tr,
    const double *deriv_x, const double *deriv_y, const double *lift,
    const double *escale, int Nq, int Ne)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  return run_z_gemm_assembly(dqdt, flux_z, D1D_tr, deriv_x, deriv_y, lift,
                             escale, Nq, Ne);
}
