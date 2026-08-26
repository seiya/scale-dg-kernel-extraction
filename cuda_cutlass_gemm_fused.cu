#include <cuda_runtime.h>
#include <cstdio>

#include "cutlass/cutlass.h"
//- Must precede the device-level GEMM headers: it specializes
//- DefaultMmaTensorOp for the K-deep f64 instruction shapes.
#include "cutlass_f64_kdeep_mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/device_kernel.h"
#include "cutlass/arch/synclog.hpp"

#include "cutlass_z_gemm_assembly.h"

// Volume GEMM tiles match the cuBLAS nsys kernels:
//   cutlass_80_tensorop_d884gemm_64x128_16x3_nn_align1
//   cutlass_80_tensorop_d884gemm_64x64_16x4_nn_align1
//   cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1
//
// The MMA instruction shape is selectable at run time (mma_shape argument):
//   0 = SM80 8x8x4   (d884, what cuBLAS picks on GB200)
//   1 = SM90 16x8x4
//   2 = SM90 16x8x8  (what cuBLAS picks on H100 for these same shapes)
//   3 = SM90 16x8x16 (needs a K=32 tile, see VolumeGemmSet)
// Shapes 0, 1 and 2 share the same threadblock tile, warp tile and stage
// count, so comparing them isolates the instruction shape.
//
// On sm_100 all four lower to the same DMMA.8x8x4 SASS instruction -- ptxas
// expands m16n8k4/8/16 into 2/4/8 of them -- so none of this can be faster
// there. On sm_90 each is a single instruction. Shapes 2 and 3 need the
// warp-level iterators in cutlass_f64_kdeep_mma.h; the stock CUTLASS 2.x ones
// only handle instructions four elements deep in K.
// See reports/sm90_mma_shape_survey.md.

//- Defined in cuda_dg_kernels_tc.cu; the stream shared by the whole CUDA path.
extern cudaStream_t dg_cuda_stream;

namespace {

using ColumnMajor = cutlass::layout::ColumnMajor;
using TensorOp = cutlass::arch::OpClassTensorOp;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using BatchedSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<double, 1, double, double>;

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;

//- One set of volume GEMMs for a given MMA instruction shape. Only InstShape,
//- ArchTag and the tile K depth vary; the M/N tiles, warp counts and stage
//- counts below are the ones the 8x8x4 path has always used.
//-
//- TileK must satisfy WarpShape::kK / InstShape::kK >= 2 and even, which
//- MmaBase asserts (mma_base.h:128,132). That is why 16x8x16 needs TileK = 32:
//- with TileK = 16 it would leave a single warp-level GEMM per stage and the
//- software pipeline degenerates. The deeper tile doubles the shared memory
//- per CTA, so its occupancy is not comparable with the other two shapes.
template <class InstShape, class ArchTag, int TileK>
struct VolumeGemmSet {
  using GemmX = cutlass::gemm::device::Gemm<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 128, TileK>, GS<32, 64, TileK>, InstShape,
      EpilogueOp, Swizzle, 3>;

  using GemmY = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 64, TileK>, GS<32, 32, TileK>, InstShape,
      EpilogueOp, BatchedSwizzle, 4>;

  using GemmZ = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 32, TileK>, GS<32, 32, TileK>, InstShape,
      EpilogueOp, BatchedSwizzle, 4>;
};

using MmaSet_884 = VolumeGemmSet<GS<8, 8, 4>, cutlass::arch::Sm80, 16>;
using MmaSet_1688 = VolumeGemmSet<GS<16, 8, 8>, cutlass::arch::Sm90, 16>;
using MmaSet_16816 = VolumeGemmSet<GS<16, 8, 16>, cutlass::arch::Sm90, 32>;
using MmaSet_1684 = VolumeGemmSet<GS<16, 8, 4>, cutlass::arch::Sm90, 16>;

int cutlass_error(const char *what, cutlass::Status st)
{
  if (st == cutlass::Status::kSuccess) {
    return 0;
  }
  std::fprintf(stderr, "%s: cutlass status %d\n", what, static_cast<int>(st));
  return static_cast<int>(st);
}

int bad_mma_shape(int mma_shape)
{
  std::fprintf(stderr, "cutlass volume gemm: unsupported mma_shape %d\n",
               mma_shape);
  return 1;
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

template <class Set>
int run_volume_gemms_xy(double *deriv_x, double *deriv_y, const double *flux_x,
                        const double *flux_y, const double *D1D,
                        const double *D1D_tr, int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const long long stride_plane = nq2;

  int st = run_gemm_nn<typename Set::GemmX>(Nq, nq2 * Ne, Nq, D1D, Nq, flux_x,
                                            Nq, deriv_x, Nq);
  if (st != 0) {
    return st;
  }

  return run_gemm_batched_nn<typename Set::GemmY>(
      Nq, Nq, Nq, flux_y, Nq, stride_plane, D1D_tr, Nq, 0, deriv_y, Nq,
      stride_plane, Nq * Ne);
}

template <class Set>
int run_volume_gemms(double *deriv_x, double *deriv_y, double *deriv_z,
                     const double *flux_x, const double *flux_y,
                     const double *flux_z, const double *D1D,
                     const double *D1D_tr, int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const int Np = nq2 * Nq;
  const long long stride_vol = Np;

  int st = run_volume_gemms_xy<Set>(deriv_x, deriv_y, flux_x, flux_y, D1D,
                                    D1D_tr, Nq, Ne);
  if (st != 0) {
    return st;
  }

  return run_gemm_batched_nn<typename Set::GemmZ>(
      nq2, Nq, Nq, flux_z, nq2, stride_vol, D1D_tr, Nq, 0, deriv_z, nq2,
      stride_vol, Ne);
}

template <class Set>
int run_z_gemm_assembly(double *dqdt, const double *flux_z, const double *D1D_tr,
                        const double *deriv_x, const double *deriv_y,
                        const double *flux_bnd, const double *lift1d,
                        const double *escale, int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const int Np = nq2 * Nq;
  const int npoint = Np * Ne;
  const long long stride_vol = Np;
  const int m = nq2;
  const int n = Nq;
  const int k = Nq;

  using GemmZ = typename Set::GemmZ;
  //- Same epilogue, with the accumulator staging tile padded so the stores are
  //- bank-conflict free. See RepadEpilogue in cutlass_z_gemm_assembly.h.
  using ZEpilogue = RepadEpilogue<typename GemmZ::GemmKernel::Epilogue, 8>;
  using Kernel = GemmBatchedDqdtAssembly<typename GemmZ::GemmKernel::Mma,
                                         ZEpilogue, BatchedSwizzle>;

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
  params.ptr_flux_bnd = flux_bnd;
  params.ptr_lift1d = lift1d;
  params.Nq = Nq;
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
    const double *D1D_tr, int Nq, int Ne, int mma_shape)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  switch (mma_shape) {
  case 0:
    return run_volume_gemms<MmaSet_884>(deriv_x, deriv_y, deriv_z, flux_x,
                                        flux_y, flux_z, D1D, D1D_tr, Nq, Ne);
  case 1:
    return run_volume_gemms<MmaSet_1684>(deriv_x, deriv_y, deriv_z, flux_x,
                                         flux_y, flux_z, D1D, D1D_tr, Nq, Ne);
  case 2:
    return run_volume_gemms<MmaSet_1688>(deriv_x, deriv_y, deriv_z, flux_x,
                                         flux_y, flux_z, D1D, D1D_tr, Nq, Ne);
  case 3:
    return run_volume_gemms<MmaSet_16816>(deriv_x, deriv_y, deriv_z, flux_x,
                                          flux_y, flux_z, D1D, D1D_tr, Nq, Ne);
  default:
    return bad_mma_shape(mma_shape);
  }
}

extern "C" int launch_volume_gemm_xy(
    double *deriv_x, double *deriv_y, const double *flux_x, const double *flux_y,
    const double *D1D, const double *D1D_tr, int Nq, int Ne, int mma_shape)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  switch (mma_shape) {
  case 0:
    return run_volume_gemms_xy<MmaSet_884>(deriv_x, deriv_y, flux_x, flux_y,
                                           D1D, D1D_tr, Nq, Ne);
  case 1:
    return run_volume_gemms_xy<MmaSet_1684>(deriv_x, deriv_y, flux_x, flux_y,
                                            D1D, D1D_tr, Nq, Ne);
  case 2:
    return run_volume_gemms_xy<MmaSet_1688>(deriv_x, deriv_y, flux_x, flux_y,
                                            D1D, D1D_tr, Nq, Ne);
  case 3:
    return run_volume_gemms_xy<MmaSet_16816>(deriv_x, deriv_y, flux_x, flux_y,
                                             D1D, D1D_tr, Nq, Ne);
  default:
    return bad_mma_shape(mma_shape);
  }
}

extern "C" int launch_z_gemm_assembly(
    double *dqdt, const double *flux_z, const double *D1D_tr,
    const double *deriv_x, const double *deriv_y, const double *flux_bnd,
    const double *lift1d, const double *escale, int Nq, int Ne, int mma_shape)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  switch (mma_shape) {
  case 0:
    return run_z_gemm_assembly<MmaSet_884>(dqdt, flux_z, D1D_tr, deriv_x,
                                           deriv_y, flux_bnd, lift1d, escale,
                                           Nq, Ne);
  case 1:
    return run_z_gemm_assembly<MmaSet_1684>(dqdt, flux_z, D1D_tr, deriv_x,
                                            deriv_y, flux_bnd, lift1d, escale,
                                            Nq, Ne);
  case 2:
    return run_z_gemm_assembly<MmaSet_1688>(dqdt, flux_z, D1D_tr, deriv_x,
                                            deriv_y, flux_bnd, lift1d, escale,
                                            Nq, Ne);
  case 3:
    return run_z_gemm_assembly<MmaSet_16816>(dqdt, flux_z, D1D_tr, deriv_x,
                                             deriv_y, flux_bnd, lift1d, escale,
                                             Nq, Ne);
  default:
    return bad_mma_shape(mma_shape);
  }
}
