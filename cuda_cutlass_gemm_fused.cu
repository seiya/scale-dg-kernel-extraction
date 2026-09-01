#include <cuda_runtime.h>
#include <cstdint>
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
#include "cutlass_y_gemm_scaleadd.h"
#include "cutlass_y_gemm_assembly.h"

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
//- Two elements per contiguous access, i.e. 16-byte loads and stores in the
//- epilogue.  Only the Nq > 64 branch takes it: there it is worth 7.2 us per
//- stage at p=127 and 17 us at p=255, while at Nq = 64 the same change makes
//- the z kernel 2.8% slower for an identical result (the same asymmetry the
//- other three z epilogue changes have, cutlass_z_gemm_assembly.h).
using EpilogueOp2 = cutlass::epilogue::thread::LinearCombination<double, 2, double, double>;


//- Epilogue output op for the x and y volume GEMMs of the fused path:
//-   D = accumulator * source
//- with the source tensor being the Escale field for that direction.  This is
//- the whole reason it exists: the z epilogue used to load Dx, Dy, Escale_x,
//- Escale_y and Escale_z, five volume tensors, and it is the kernel with the
//- least room -- 73% SM throughput at 12.5% occupancy -- while the x and y
//- GEMMs sit at 88-90% SM with 6-10% DRAM.  Weighting there and reading three
//- tensors here is worth 19.9 us per stage (measured by ablation).
//-
//- It has to be an OutputOp rather than a hand-written epilogue: replacing
//- CUTLASS's stock epilogue with a hand-rolled one on the y GEMM costs 72 us
//- per stage all by itself, which is four times what the whole move can win.
//- See reports/p255_gap_study.md.
template <int kCountV>
class PointwiseScaleV {
public:
  using ElementOutput = double;
  using ElementSource = double;
  using ElementAccumulator = double;
  using ElementCompute = double;
  using ElementC = double;
  using ElementD = double;

  static int const kCount = kCountV;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentSource = cutlass::Array<ElementSource, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;

  struct Params {
    CUTLASS_HOST_DEVICE Params() {}
  };

  CUTLASS_HOST_DEVICE
  explicit PointwiseScaleV(Params const & = Params()) {}

  CUTLASS_HOST_DEVICE
  explicit PointwiseScaleV(Params const &, int) {}

  //- The source is the Escale field, so it is always needed.
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }

  //- Split-K would make the partial sums meet after the scaling, which is not
  //- what this op means.  The volume GEMMs never split K.
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentSource const &source) const
  {
    FragmentOutput out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      out[i] = accumulator[i] * source[i];
    }
    return out;
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const
  {
    FragmentOutput out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      out[i] = accumulator[i];
    }
    return out;
  }
};


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

  //- Three stages, not four: see GemmYScaleShallow below.  This epilogue has
  //- no second source, so unlike the fused y GEMM it wins at Nq = 64 too.
  using GemmY = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 64, TileK>, GS<32, 32, TileK>, InstShape,
      EpilogueOp, BatchedSwizzle, 3>;

  //- Nq=32 tile: four 16x16 warps own the whole matrix. Three stages are
  //- enough for the two TileK=16 iterations and raise occupancy over the
  //- five-stage cuBLAS kernel selected for this shape.
  using GemmY32 = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<32, 32, TileK>, GS<16, 16, TileK>, InstShape,
      EpilogueOp, BatchedSwizzle, 3>;

  //- The x and y GEMMs of the fused path, with Escale folded into the stock
  //- epilogue through PointwiseScale.  Same tiles as GemmX / GemmY.
  using GemmXScale = cutlass::gemm::device::Gemm<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 128, TileK>, GS<32, 64, TileK>, InstShape,
      PointwiseScaleV<2>, Swizzle, 3>;

  using GemmYScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 64, TileK>, GS<32, 32, TileK>, InstShape,
      PointwiseScaleV<2>, BatchedSwizzle, 4>;

  //- Same y GEMM with a three-deep pipeline instead of four.  These GEMMs run
  //- at 95% of the SM throughput roof, so they are issue bound and not latency
  //- bound, and the fourth stage only pays for a longer prologue: it costs
  //- 0.80% more instructions and buys nothing.  The freed 16 KB of shared does
  //- not raise occupancy (three CTAs either way, registers cap it), so the win
  //- is the shorter pipeline alone.  Nq = 64 is the exception and keeps four
  //- (reports/p511_gap_study.md section 12).  At Nq >= 512 the x GEMM shares
  //- this tile and takes the same pipeline; five stages there is +1.3% and
  //- TileK = 32 is +1.6%, both because the extra shared memory drops the SM
  //- to two CTAs (reports/p575_gap_study.md sections 11.13 and 11.14).
  using GemmYScaleShallow = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 64, TileK>, GS<32, 32, TileK>, InstShape,
      PointwiseScaleV<2>, BatchedSwizzle, 3>;

  using GemmZ = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 32, TileK>, GS<32, 32, TileK>, InstShape,
      EpilogueOp, BatchedSwizzle, 4>;

  //- Same z GEMM with 16-byte epilogue accesses, for the Nq > 64 branch.
  using GemmZWide = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 32, TileK>, GS<32, 32, TileK>, InstShape,
      EpilogueOp2, BatchedSwizzle, 4>;
};

using MmaSet_884 = VolumeGemmSet<GS<8, 8, 4>, cutlass::arch::Sm80, 16>;
using MmaSet_1688 = VolumeGemmSet<GS<16, 8, 8>, cutlass::arch::Sm90, 16>;
using MmaSet_16816 = VolumeGemmSet<GS<16, 8, 16>, cutlass::arch::Sm90, 32>;
using MmaSet_1684 = VolumeGemmSet<GS<16, 8, 4>, cutlass::arch::Sm90, 16>;

// Nq=8 makes the generic tiles mostly predicated. CUTLASS's FP64 TensorOp
// multistage iterator needs at least two warps; the measured winners are
// 32x64 for the 8x8 y batches and 16x32 for the 64x8 z/assembly GEMM.
struct VolumeGemmSetP7 : VolumeGemmSet<GS<8, 8, 4>, cutlass::arch::Sm80, 16> {
  using GemmY = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, cutlass::arch::Sm80, GS<32, 64, 16>, GS<32, 32, 16>,
      GS<8, 8, 4>, EpilogueOp, BatchedSwizzle, 3>;
  using GemmYScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, cutlass::arch::Sm80, GS<32, 64, 16>, GS<32, 32, 16>,
      GS<8, 8, 4>, PointwiseScaleV<1>, BatchedSwizzle, 3>;
  using GemmZ = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, cutlass::arch::Sm80, GS<16, 32, 16>, GS<16, 16, 16>,
      GS<8, 8, 4>, EpilogueOp, BatchedSwizzle, 3>;
  //- Same tile with 16-byte epilogue accesses.  Only the low-order study of
  //- reports/p7_gemm_fused.md section 13 instantiates it; the mainloop is the
  //- one GemmZ has, so GEMM_CUTE and GEMM_FUSED still share it.
  using GemmZWide = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, cutlass::arch::Sm80, GS<16, 32, 16>, GS<16, 16, 16>,
      GS<8, 8, 4>, EpilogueOp2, BatchedSwizzle, 3>;
};

using P7MmaSet_884 = VolumeGemmSetP7;

//- Order-specialized x volume GEMM tiles.  The x GEMM is Nq x (nq2*Ne) x Nq
//- with a column-major C, so device::Gemm runs the transposed problem: the
//- threadblock's M dimension covers the long nq2*Ne axis and its N dimension
//- covers Nq.  The generic GemmX tile is 64x128, so at Nq = 8 120 of its 128
//- N columns are predicated away, at Nq = 16 112 of them, at Nq = 32 96 and
//- at Nq = 64 still half -- which is why cuBLAS held this GEMM at every order
//- up to 64.  These candidates keep the M extent and shrink N to Nq (16 is
//- the floor, see below).  Whichever one measures fastest is used by
//- GEMM_CUTE and GEMM_FUSED alike; only the Escale-weighted variant is
//- fused-only, and it differs from the plain one in the epilogue output op
//- alone.
template <int TbM, int TbN, int WM, int WN, int TileK = 16, int Stages = 3>
struct XTile {
  using GemmX = cutlass::gemm::device::Gemm<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, cutlass::arch::Sm80, GS<TbM, TbN, TileK>, GS<WM, WN, TileK>,
      GS<8, 8, 4>, EpilogueOp, Swizzle, Stages>;
  using GemmXScale = cutlass::gemm::device::Gemm<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, cutlass::arch::Sm80, GS<TbM, TbN, TileK>, GS<WM, WN, TileK>,
      GS<8, 8, 4>, PointwiseScaleV<1>, Swizzle, Stages>;
};

//- N = 8 does not build: the FP64 multistage B iterator's thread map
//- degenerates to zero contiguous iterations for a 16x8 tile with 128
//- threads, so 16 is the narrowest usable N and half of it is predicated at
//- Nq = 8.  Same reason the p=7 y and z tiles are two warps wide.
using P7XTile0 = XTile<64, 16, 32, 16>;
using P7XTile1 = XTile<128, 16, 64, 16>;
using P7XTile2 = XTile<32, 16, 16, 16>;
using P7XTile3 = XTile<256, 16, 64, 16>;
using P7XTile4 = XTile<128, 16, 32, 16>;
using P7XTile5 = XTile<64, 32, 32, 32>;
using P7XTile6 = XTile<64, 16, 32, 16, 16, 4>;
using P7XTile7 = XTile<64, 16, 32, 16, 16, 5>;
using P7XTile8 = XTile<64, 16, 32, 16, 32, 3>;
using P7XTile9 = XTile<64, 16, 16, 16, 16, 3>;
using P7XTile12 = XTile<32, 16, 16, 16, 16, 3>;
using P7XTile13 = XTile<128, 16, 16, 16, 16, 3>;
using P7XTile14 = XTile<64, 16, 16, 16, 16, 4>;
using P7XTile15 = XTile<256, 16, 32, 16, 16, 3>;
//- Indices 10 and 11 are absent on purpose: a one-warp threadblock (32x16
//- with a 32x16 warp) and TileK = 8 both fail the CUTLASS static assertions
//- for this iterator, so they are not options at Nq = 8.  SCALE_DG_XTILE=10
//- or 11 therefore reports a bad tile rather than silently running something
//- else.

//- The same sweep at Nq = 16, 32 and 64.  N is Nq exactly, so nothing is
//- predicated in the N direction any more and K is covered in Nq/TileK
//- iterations.  The candidates vary the long-axis extent TbM, the warp
//- partition, TileK and the pipeline depth; the shared tile that measures
//- fastest is the one both GEMM paths use at that order.
using P15XTile0 = XTile<64, 16, 32, 16, 16, 3>;
using P15XTile1 = XTile<128, 16, 64, 16, 16, 3>;
using P15XTile2 = XTile<32, 16, 16, 16, 16, 3>;
using P15XTile3 = XTile<256, 16, 64, 16, 16, 3>;
using P15XTile4 = XTile<128, 16, 32, 16, 16, 3>;
using P15XTile5 = XTile<64, 16, 16, 16, 16, 3>;
using P15XTile6 = XTile<64, 16, 32, 16, 16, 4>;
using P15XTile7 = XTile<64, 16, 16, 16, 16, 4>;
using P15XTile8 = XTile<128, 16, 16, 16, 16, 3>;
using P15XTile9 = XTile<256, 16, 32, 16, 16, 3>;
using P15XTile10 = XTile<64, 32, 32, 32, 16, 3>;
using P15XTile11 = XTile<64, 16, 32, 16, 32, 3>;

using P31XTile0 = XTile<64, 32, 32, 32, 16, 3>;
using P31XTile1 = XTile<64, 32, 16, 32, 16, 3>;
using P31XTile2 = XTile<64, 32, 32, 16, 16, 3>;
using P31XTile3 = XTile<128, 32, 64, 32, 16, 3>;
using P31XTile4 = XTile<128, 32, 32, 32, 16, 3>;
using P31XTile5 = XTile<256, 32, 64, 32, 16, 3>;
using P31XTile6 = XTile<32, 32, 16, 32, 16, 3>;
using P31XTile7 = XTile<32, 32, 16, 16, 16, 3>;
using P31XTile8 = XTile<64, 32, 32, 32, 16, 4>;
using P31XTile9 = XTile<64, 32, 32, 32, 32, 3>;
using P31XTile10 = XTile<64, 32, 32, 32, 16, 5>;
using P31XTile11 = XTile<128, 32, 32, 32, 16, 4>;

using P63XTile0 = XTile<64, 64, 32, 32, 16, 3>;
using P63XTile1 = XTile<64, 64, 32, 64, 16, 3>;
using P63XTile2 = XTile<64, 64, 64, 32, 16, 3>;
using P63XTile3 = XTile<128, 64, 64, 32, 16, 3>;
using P63XTile4 = XTile<128, 64, 32, 32, 16, 3>;
using P63XTile5 = XTile<32, 64, 32, 32, 16, 3>;
using P63XTile6 = XTile<32, 64, 16, 32, 16, 3>;
using P63XTile7 = XTile<64, 64, 32, 32, 16, 4>;
using P63XTile8 = XTile<64, 64, 32, 32, 32, 3>;
using P63XTile9 = XTile<256, 64, 64, 32, 16, 3>;
using P63XTile10 = XTile<64, 64, 16, 32, 16, 3>;
using P63XTile11 = XTile<64, 32, 32, 32, 16, 3>;

//- Nq=16 leaves the generic tiles as predicated as Nq=8 does: the y GEMM is
//- 16x16x16 per batch against a 64x64 tile, and the z / z-assembly GEMM is
//- 256x16x16 against a 64x32 tile whose N half is predicated away.  Same
//- reasoning as VolumeGemmSetP7; the shapes below are the measured winners
//- (reports/p15_gap_study.md).  Both GEMM_CUTE and GEMM_FUSED take them, so
//- the two paths keep sharing one volume-GEMM mainloop.
struct VolumeGemmSetP15 : VolumeGemmSet<GS<8, 8, 4>, cutlass::arch::Sm80, 16> {
  using GemmY = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, cutlass::arch::Sm80, GS<16, 32, 16>,
      GS<16, 16, 16>, GS<8, 8, 4>, EpilogueOp, BatchedSwizzle, 3>;
  using GemmYScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, cutlass::arch::Sm80, GS<16, 32, 16>,
      GS<16, 16, 16>, GS<8, 8, 4>, PointwiseScaleV<1>, BatchedSwizzle, 3>;
  using GemmZ = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, cutlass::arch::Sm80, GS<16, 64, 16>,
      GS<16, 32, 16>, GS<8, 8, 4>, EpilogueOp, BatchedSwizzle, 3>;
  using GemmZWide = GemmZ;
};

using P15MmaSet_884 = VolumeGemmSetP15;

int cutlass_error(const char *what, cutlass::Status st)
{
  if (st == cutlass::Status::kSuccess) {
    return 0;
  }
  std::fprintf(stderr, "%s: cutlass status %d\n", what, static_cast<int>(st));
  return static_cast<int>(st);
}

// CUDA grid.z max is 65535.  CUTLASS GemmBatchedIdentityThreadblockSwizzle
// stores batch as `count % 65536`, so Nq*Ne == 65536 (p=15, Ne=16^3)
// would launch with grid.z = 0 (cutlass status 7).  The batched kernels
// already walk batch_idx += gridDim.z, so capping z is enough.
constexpr int kCudaMaxGridZ = 65535;

int batched_grid_z(int batch_count)
{
  return batch_count > kCudaMaxGridZ ? kCudaMaxGridZ : batch_count;
}

cutlass::gemm::GemmCoord cap_batched_grid(cutlass::gemm::GemmCoord tiled,
                                           int batch_count)
{
  return cutlass::gemm::GemmCoord(tiled.m(), tiled.n(),
                                   batched_grid_z(batch_count));
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

//- One launch for any batch count: the device-level operator cannot express
//- batch > 65535 (it copies batch into grid_tiled_shape.k() via `% 65536`),
//- but kernel::GemmBatched itself walks `batch_idx += gridDim.z`, so building
//- Params by hand with the full batch_count and a capped grid.z covers every
//- batch in a single kernel.  Chunking instead costs one launch per 65535
//- batches plus, at p=15 (batch = Nq*Ne = 65536 exactly), a second launch that
//- carries a single 16x16x16 plane.
template <class GemmBatched>
int run_gemm_batched_nn_capped(int m, int n, int k, double const *A, int lda,
                               long long strideA, double const *B, int ldb,
                               long long strideB, double *C, int ldc,
                               long long strideC, int batch)
{
  //- Same mainloop, with the epilogue's accumulator staging tile padded by 8
  //- the way the z assembly pads it (RepadEpilogue, cutlass_z_gemm_assembly.h).
  //- Without it the y GEMM takes 2.18 M shared-store bank conflicts (ncu job
  //- 74635); pad 4 and 16 are neutral, 8 is worth -0.6% at p=15 and -4.1% at
  //- p=7 and does not move p=63.
  using Kernel = cutlass::gemm::kernel::GemmBatched<
      typename GemmBatched::GemmKernel::Mma,
      RepadEpilogue<typename GemmBatched::GemmKernel::Epilogue, 8>,
      BatchedSwizzle>;

  typename GemmBatched::Arguments args({m, n, k}, {A, lda}, strideA, {B, ldb},
                                       strideB, {C, ldc}, strideC, {C, ldc},
                                       strideC, {1.0, 0.0}, batch);
  const cutlass::Status can = GemmBatched::can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("batched capped can_implement", can);
  }

  auto uargs = GemmBatched::to_underlying_arguments(args);
  BatchedSwizzle swizzle;
  cutlass::gemm::GemmCoord grid_shape = cap_batched_grid(
      swizzle.get_tiled_shape(uargs.problem_size,
                              {GemmBatched::ThreadblockShape::kM,
                               GemmBatched::ThreadblockShape::kN,
                               GemmBatched::ThreadblockShape::kK},
                              uargs.batch_count),
      batch);

  typename Kernel::Params params(
      uargs.problem_size, grid_shape, uargs.ref_A.non_const_ref(), uargs.stride_A,
      uargs.ref_B.non_const_ref(), uargs.stride_B, uargs.ref_C.non_const_ref(),
      uargs.stride_C, uargs.ref_D, uargs.stride_D, uargs.epilogue, batch);

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
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

template <class GemmBatched>
int run_gemm_batched_nn(int m, int n, int k, double const *A, int lda,
                        long long strideA, double const *B, int ldb,
                        long long strideB, double *C, int ldc, long long strideC,
                        int batch)
{
  return run_gemm_batched_nn_capped<GemmBatched>(m, n, k, A, lda, strideA, B,
                                                 ldb, strideB, C, ldc, strideC,
                                                 batch);
}

template <class Gemm>
int run_gemm_nn_scaled(int m, int n, int k, double const *A, int lda,
                       double const *B, int ldb, double const *C, int ldc,
                       double *D, int ldd)
{
  Gemm gemm_op;
  typename Gemm::Arguments args({m, n, k}, {A, lda}, {B, ldb}, {C, ldc}, {D, ldd},
                                typename Gemm::EpilogueOutputOp::Params());
  const cutlass::Status can = gemm_op.can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("scaled can_implement", can);
  }
  return cutlass_error("scaled gemm", gemm_op(args, nullptr, dg_cuda_stream));
}

//- Same GEMM as run_gemm_nn_scaled with the epilogue's accumulator staging
//- tile padded by 8 (RepadEpilogue, cutlass_z_gemm_assembly.h).  The x volume
//- GEMM is the last one still on the stock epilogue, and ncu job 74732 counts
//- 32.6 M shared-store bank conflicts in it -- the same 2-way conflict the
//- batched launcher and the z assembly already pad away.  The device-level
//- operator cannot take a different epilogue, so Params is built here.
template <class Gemm>
int run_gemm_nn_scaled_repad(int m, int n, int k, double const *A, int lda,
                             double const *B, int ldb, double const *C, int ldc,
                             double *D, int ldd)
{
  using Kernel = cutlass::gemm::kernel::Gemm<
      typename Gemm::GemmKernel::Mma,
      RepadEpilogue<typename Gemm::GemmKernel::Epilogue, 8>, Swizzle, false>;

  typename Gemm::Arguments args({m, n, k}, {A, lda}, {B, ldb}, {C, ldc},
                                {D, ldd},
                                typename Gemm::EpilogueOutputOp::Params());
  const cutlass::Status can = Gemm::can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("scaled repad can_implement", can);
  }

  //- The column-major-C device operator runs the transposed problem, so build
  //- Params from its underlying arguments rather than from ours.
  auto uargs = Gemm::to_underlying_arguments(args);
  Swizzle swizzle;
  cutlass::gemm::GemmCoord grid_shape = swizzle.get_tiled_shape(
      uargs.problem_size,
      {Gemm::ThreadblockShape::kM, Gemm::ThreadblockShape::kN,
       Gemm::ThreadblockShape::kK},
      1);

  typename Kernel::Params params{uargs.problem_size,
                                 grid_shape,
                                 uargs.ref_A.non_const_ref(),
                                 uargs.ref_B.non_const_ref(),
                                 uargs.ref_C.non_const_ref(),
                                 uargs.ref_D,
                                 uargs.epilogue,
                                 nullptr};

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
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

//- Same launch as run_gemm_batched_nn_capped, for the two-tensor epilogue that
//- run_gemm_batched_nn_scaled drives: full batch_count with a capped grid.z,
//- and the accumulator staging tile padded by 8 so the epilogue's shared
//- stores are conflict free (RepadEpilogue, cutlass_z_gemm_assembly.h).  The
//- Nq >= 512 x GEMM is the only caller that needs the pad; without it that
//- kernel takes the same 2-way store conflict the y GEMM used to.
template <class GemmBatched>
int run_gemm_batched_nn_scaled_capped(int m, int n, int k, double const *A,
                                      int lda, long long strideA,
                                      double const *B, int ldb,
                                      long long strideB, double const *C,
                                      int ldc, long long strideC, double *D,
                                      int ldd, long long strideD, int batch)
{
  using Kernel = cutlass::gemm::kernel::GemmBatched<
      typename GemmBatched::GemmKernel::Mma,
      RepadEpilogue<typename GemmBatched::GemmKernel::Epilogue, 8>,
      BatchedSwizzle>;

  typename GemmBatched::Arguments args(
      {m, n, k}, {A, lda}, strideA, {B, ldb}, strideB, {C, ldc}, strideC,
      {D, ldd}, strideD, typename GemmBatched::EpilogueOutputOp::Params(),
      batch);
  const cutlass::Status can = GemmBatched::can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("scaled batched capped can_implement", can);
  }

  auto uargs = GemmBatched::to_underlying_arguments(args);
  BatchedSwizzle swizzle;
  cutlass::gemm::GemmCoord grid_shape = cap_batched_grid(
      swizzle.get_tiled_shape(uargs.problem_size,
                              {GemmBatched::ThreadblockShape::kM,
                               GemmBatched::ThreadblockShape::kN,
                               GemmBatched::ThreadblockShape::kK},
                              uargs.batch_count),
      batch);

  typename Kernel::Params params(
      uargs.problem_size, grid_shape, uargs.ref_A.non_const_ref(), uargs.stride_A,
      uargs.ref_B.non_const_ref(), uargs.stride_B, uargs.ref_C.non_const_ref(),
      uargs.stride_C, uargs.ref_D, uargs.stride_D, uargs.epilogue, batch);

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
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

template <class GemmBatched>
int run_gemm_batched_nn_scaled(int m, int n, int k, double const *A, int lda,
                               long long strideA, double const *B, int ldb,
                               long long strideB, double const *C, int ldc,
                               long long strideC, double *D, int ldd,
                               long long strideD, int batch)
{
  return run_gemm_batched_nn_scaled_capped<GemmBatched>(
      m, n, k, A, lda, strideA, B, ldb, strideB, C, ldc, strideC, D, ldd,
      strideD, batch);
}

//- y GEMM whose epilogue takes two sources: D = Escale_y * acc + deriv_x.
//- kMulAddend multiplies the addend by Escale_x (cuBLAS x has no epilogue).
//- Same tiles, warps and mainloop as GemmYScale; only the epilogue differs.
//- kShallowPipe picks GemmYScaleShallow, which differs only in stage count.
//- cutlass_y_gemm_scaleadd.h says why deriv_x is read here.
template <class Set, bool kMulAddend, bool kShallowPipe = false>
int run_volume_gemm_y_scaleadd(double *deriv_xy, const double *flux_y,
                               const double *D1D_tr, const double *escale_y,
                               const double *deriv_x, const double *escale_x,
                               int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const long long stride_plane = nq2;

  using GemmY = typename cutlass::platform::conditional<
      kShallowPipe, typename Set::GemmYScaleShallow,
      typename Set::GemmYScale>::type;
  //- Same accumulator repad the z assembly and the plain batched launcher use
  //- (RepadEpilogue, cutlass_z_gemm_assembly.h).  The stock epilogue stages the
  //- accumulators unpadded and takes a 2-way shared-store conflict on every
  //- phase.
  using Kernel = GemmBatchedScaleAdd<typename GemmY::GemmKernel::Mma,
                                     RepadEpilogue<typename GemmY::GemmKernel::Epilogue, 8>,
                                     BatchedSwizzle, kMulAddend>;

  cutlass::TensorRef<double const, ColumnMajor> ref_A(flux_y, Nq);
  cutlass::TensorRef<double const, ColumnMajor> ref_B(D1D_tr, Nq);
  cutlass::TensorRef<double, ColumnMajor> ref_D(deriv_xy, Nq);

  typename GemmY::Arguments gemm_args({Nq, Nq, Nq}, ref_A, stride_plane, ref_B, 0,
                                      ref_D, stride_plane, ref_D, stride_plane,
                                      typename GemmY::EpilogueOutputOp::Params(), Nq * Ne);
  cutlass::Status can = GemmY::can_implement(gemm_args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("y-scaleadd can_implement", can);
  }

  auto uargs = GemmY::to_underlying_arguments(gemm_args);
  BatchedSwizzle swizzle;
  cutlass::gemm::GemmCoord grid_shape = cap_batched_grid(
      swizzle.get_tiled_shape(uargs.problem_size,
                              {GemmY::ThreadblockShape::kM, GemmY::ThreadblockShape::kN,
                               GemmY::ThreadblockShape::kK},
                              uargs.batch_count),
      Nq * Ne);

  typename Kernel::Params params;
  params.gemm = typename Kernel::BaseKernel::Params(
      uargs.problem_size, grid_shape, uargs.ref_A.non_const_ref(), uargs.stride_A,
      uargs.ref_B.non_const_ref(), uargs.stride_B, uargs.ref_C.non_const_ref(),
      uargs.stride_C, uargs.ref_D, uargs.stride_D, uargs.epilogue, uargs.batch_count);
  params.ptr_scale = escale_y;
  params.ptr_add = deriv_x;
  params.ptr_mul = escale_x;
  params.stride_scale = stride_plane;
  params.stride_add = stride_plane;
  params.stride_mul = stride_plane;

  dim3 grid = swizzle.get_grid_shape(grid_shape);
  dim3 block(Kernel::kThreadCount, 1, 1);
  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (attr != cudaSuccess) {
      return 1;
    }
  }
  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel><<<grid, block, smem_size, dg_cuda_stream>>>(params);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

template <class Set>
int run_volume_gemm_x_scale(double *deriv_x, const double *flux_x,
                            const double *D1D, const double *escale, int Nq,
                            int Ne)
{
  const int nq2 = Nq * Nq;
  //- p=255 tried batched x and lost 0.7% (p255_gap_study.md §10.4).  At
  //- Nq>=512 the single 64x128 GEMM is 84% of peak while the same-FLOP y
  //- GEMM (64x64 batched) is 95%.  Use y's tile for both CUTE and fused x.
  if (Nq >= 512) {
    const long long stride = nq2;
    return run_gemm_batched_nn_scaled<typename Set::GemmYScaleShallow>(
        Nq, Nq, Nq, D1D, Nq, 0, flux_x, Nq, stride, escale, Nq, stride, deriv_x,
        Nq, stride, Nq * Ne);
  }
  return run_gemm_nn_scaled_repad<typename Set::GemmXScale>(
      Nq, nq2 * Ne, Nq, D1D, Nq, flux_x, Nq, escale, Nq, deriv_x, Nq);
}

template <class Set>
int run_volume_gemms_xy(double *deriv_x, double *deriv_y, const double *flux_x,
                        const double *flux_y, const double *D1D,
                        const double *D1D_tr, const double *escale, int Nq,
                        int Ne)
{
  const int nq2 = Nq * Nq;
  const int npoint = nq2 * Nq * Ne;

  int st = run_volume_gemm_x_scale<Set>(deriv_x, flux_x, D1D, escale, Nq, Ne);
  if (st != 0) {
    return st;
  }

  //- deriv_y comes out holding Escale_y * D(flux_y) + deriv_x, so the z
  //- epilogue reads one volume tensor instead of two.  See
  //- cutlass_y_gemm_scaleadd.h.
  //- Nq > 64 only: the three-stage y pipeline wins from Nq = 128 up and loses
  //- at Nq = 64, so the Nq <= 64 branch above keeps the four-stage GemmYScale.
  return run_volume_gemm_y_scaleadd<Set, false, true>(
      deriv_y, flux_y, D1D_tr, escale + npoint, deriv_x, nullptr, Nq, Ne);
}

//- The Nq <= 64 fused branch still uses cuBLAS for x (no Escale epilogue).
//- y then does Ey*acc + Ex*Dx in one epilogue so z can run kWeighted without
//- a separate scale kernel.  GemmZWide stays off: at Nq = 64 it is +2.8%.
template <class Set>
int run_volume_gemm_y(double *deriv_y, const double *flux_y,
                      const double *D1D_tr, int Nq, int Ne)
{
  const long long stride_plane = Nq * Nq;

  return run_gemm_batched_nn<typename Set::GemmY>(
      Nq, Nq, Nq, flux_y, Nq, stride_plane, D1D_tr, Nq, 0, deriv_y, Nq,
      stride_plane, Nq * Ne);
}

template <class Set>
int run_volume_gemm_y32(double *deriv_y, const double *flux_y,
                        const double *D1D_tr, int Nq, int Ne)
{
  const long long stride_plane = Nq * Nq;
  return run_gemm_batched_nn<typename Set::GemmY32>(
      Nq, Nq, Nq, flux_y, Nq, stride_plane, D1D_tr, Nq, 0, deriv_y, Nq,
      stride_plane, Nq * Ne);
}

template <class Set>
int run_volume_gemm_x(double *deriv_x, const double *flux_x, const double *D1D,
                      int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  if (Nq >= 512) {
    const long long stride = nq2;
    return run_gemm_batched_nn<typename Set::GemmY>(
        Nq, Nq, Nq, D1D, Nq, 0, flux_x, Nq, stride, deriv_x, Nq, stride,
        Nq * Ne);
  }
  return run_gemm_nn<typename Set::GemmX>(Nq, nq2 * Ne, Nq, D1D, Nq, flux_x,
                                          Nq, deriv_x, Nq);
}

//- The x volume GEMM on an order-specialized CUTLASS tile.  escale = nullptr is the unweighted
//- form both GEMM paths share; a non-null escale folds Escale_x into the
//- epilogue and belongs to GEMM_FUSED's fusion package.  Both go through the
//- repadded epilogue, so the two differ only in the output op.
template <class Tile>
int run_volume_gemm_x_tiled(double *deriv_x, const double *flux_x,
                         const double *D1D, const double *escale, int Nq,
                         int Ne)
{
  const int n = Nq * Nq * Ne;
  if (escale == nullptr) {
    //- LinearCombination with beta = 0 never reads C, so passing deriv_x as
    //- the source is inert; it keeps one launch helper for both forms.
    return run_gemm_nn_scaled_repad<typename Tile::GemmX>(
        Nq, n, Nq, D1D, Nq, flux_x, Nq, deriv_x, Nq, deriv_x, Nq);
  }
  return run_gemm_nn_scaled_repad<typename Tile::GemmXScale>(
      Nq, n, Nq, D1D, Nq, flux_x, Nq, escale, Nq, deriv_x, Nq);
}

template <class Set>
int run_volume_gemm_z(double *deriv_z, const double *flux_z,
                      const double *D1D_tr, int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const long long stride_vol = static_cast<long long>(nq2) * Nq;
  return run_gemm_batched_nn<typename Set::GemmZ>(
      nq2, Nq, Nq, flux_z, nq2, stride_vol, D1D_tr, Nq, 0, deriv_z, nq2,
      stride_vol, Ne);
}

template <class Set>
int run_volume_gemms(double *deriv_x, double *deriv_y, double *deriv_z,
                     const double *flux_x, const double *flux_y,
                     const double *flux_z, const double *D1D,
                     const double *D1D_tr, int Nq, int Ne)
{
  int st = run_volume_gemm_x<Set>(deriv_x, flux_x, D1D, Nq, Ne);
  if (st != 0) {
    return st;
  }
  st = run_volume_gemm_y<Set>(deriv_y, flux_y, D1D_tr, Nq, Ne);
  if (st != 0) {
    return st;
  }
  return run_volume_gemm_z<Set>(deriv_z, flux_z, D1D_tr, Nq, Ne);
}

template <class Set, bool kWeighted, bool kWide = kWeighted,
          bool kAffine = kWeighted, bool kPaired = kWeighted,
          bool kXWeighted = false>
int run_z_gemm_assembly(double *dqdt, const double *flux_z, const double *D1D_tr,
                        const double *deriv_x, const double *deriv_y,
                        const double *flux_bnd, const double *lift1d,
                        const double *lift_zpair, const double *escale, int Nq,
                        int Ne)
{
  const int nq2 = Nq * Nq;
  const int Np = nq2 * Nq;
  // Pointer arithmetic is 64-bit even when each kernel-local index remains in
  // the validated 32-bit range.  Promote before multiplying so large batches
  // cannot wrap while locating the y/z Escale components.
  const std::int64_t npoint = std::int64_t{Np} * Ne;
  const long long stride_vol = Np;
  const int m = nq2;
  const int n = Nq;
  const int k = Nq;

  using GemmZ = typename cutlass::platform::conditional<
      kWide, typename Set::GemmZWide, typename Set::GemmZ>::type;
  //- Same epilogue, with the accumulator staging tile padded so the stores are
  //- bank-conflict free. See RepadEpilogue in cutlass_z_gemm_assembly.h.
  using ZEpilogue = RepadEpilogue<typename GemmZ::GemmKernel::Epilogue, 8>;
  using Kernel = GemmBatchedDqdtAssembly<typename GemmZ::GemmKernel::Mma,
                                         ZEpilogue, BatchedSwizzle, kWeighted,
                                         kAffine, kPaired, kXWeighted>;

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
  cutlass::gemm::GemmCoord grid_shape = cap_batched_grid(
      swizzle.get_tiled_shape(uargs.problem_size,
                              {GemmZ::ThreadblockShape::kM, GemmZ::ThreadblockShape::kN,
                               GemmZ::ThreadblockShape::kK},
                              uargs.batch_count),
      Ne);

  typename Kernel::Params params;
  params.gemm = typename Kernel::BaseKernel::Params(
      uargs.problem_size, grid_shape, uargs.ref_A.non_const_ref(), uargs.stride_A,
      uargs.ref_B.non_const_ref(), uargs.stride_B, uargs.ref_C.non_const_ref(),
      uargs.stride_C, uargs.ref_D, uargs.stride_D, uargs.epilogue,
      uargs.batch_count);
  //- On the weighted branch the y GEMM already added deriv_x into deriv_y, so
  //- the tensor the epilogue reads as "dx" is the merged one.
  params.ptr_dx = kWeighted ? deriv_y : deriv_x;
  params.ptr_dy = deriv_y;
  params.ptr_flux_bnd = flux_bnd;
  params.ptr_lift1d = lift1d;
  params.ptr_lift_zpair = lift_zpair;
  params.Nq = Nq;
  //- Only read when kWeighted is false, but always valid: a null iterator base
  //- would still be dereferenced by the predicated tile iterator.
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

//- The same fusion package on the y GEMM instead of the z GEMM.  x and z run
//- first as plain unweighted GEMMs into deriv_x and a scratch deriv_z, and this
//- kernel -- whose mainloop is the one GEMM_CUTE runs for y -- applies
//- Escale_x/y/z, adds the six-face lift and writes dqdt.  See
//- cutlass_y_gemm_assembly.h for the geometry.
template <class GemmYT, bool kXWeighted>
int run_y_gemm_assembly(double *dqdt, const double *flux_y, const double *D1D_tr,
                        const double *deriv_x, const double *deriv_z,
                        const double *flux_bnd, const double *lift1d,
                        const double *escale, int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const int Np = nq2 * Nq;
  const std::int64_t npoint = std::int64_t{Np} * Ne;
  const long long stride_plane = nq2;
  const int m = Nq;
  const int n = Nq;
  const int k = Nq;
  const int batch = Nq * Ne;

  using YEpilogue = RepadEpilogue<typename GemmYT::GemmKernel::Epilogue, 8>;
  using Kernel = GemmBatchedDqdtAssemblyY<typename GemmYT::GemmKernel::Mma,
                                          YEpilogue, BatchedSwizzle, kXWeighted>;

  cutlass::TensorRef<double const, ColumnMajor> ref_A(flux_y, m);
  cutlass::TensorRef<double const, ColumnMajor> ref_B(D1D_tr, k);
  cutlass::TensorRef<double, ColumnMajor> ref_D(dqdt, m);

  typename GemmYT::Arguments gemm_args({m, n, k}, ref_A, stride_plane, ref_B, 0,
                                       ref_D, stride_plane, ref_D, stride_plane,
                                       {1.0, 0.0}, batch);
  cutlass::Status can = GemmYT::can_implement(gemm_args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("y-assembly can_implement", can);
  }

  auto uargs = GemmYT::to_underlying_arguments(gemm_args);
  BatchedSwizzle swizzle;
  cutlass::gemm::GemmCoord grid_shape = cap_batched_grid(
      swizzle.get_tiled_shape(uargs.problem_size,
                              {GemmYT::ThreadblockShape::kM, GemmYT::ThreadblockShape::kN,
                               GemmYT::ThreadblockShape::kK},
                              uargs.batch_count),
      batch);

  typename Kernel::Params params;
  params.gemm = typename Kernel::BaseKernel::Params(
      uargs.problem_size, grid_shape, uargs.ref_A.non_const_ref(), uargs.stride_A,
      uargs.ref_B.non_const_ref(), uargs.stride_B, uargs.ref_C.non_const_ref(),
      uargs.stride_C, uargs.ref_D, uargs.stride_D, uargs.epilogue, batch);
  params.ptr_dx = deriv_x;
  params.ptr_dz = deriv_z;
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

extern "C" int launch_volume_gemm_x(double *deriv_x, const double *flux_x,
                                    const double *D1D, int Nq, int Ne,
                                    int mma_shape)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  switch (mma_shape) {
  case 0:
    return run_volume_gemm_x<MmaSet_884>(deriv_x, flux_x, D1D, Nq, Ne);
  case 1:
    return run_volume_gemm_x<MmaSet_1684>(deriv_x, flux_x, D1D, Nq, Ne);
  case 2:
    return run_volume_gemm_x<MmaSet_1688>(deriv_x, flux_x, D1D, Nq, Ne);
  case 3:
    return run_volume_gemm_x<MmaSet_16816>(deriv_x, flux_x, D1D, Nq, Ne);
  default:
    return bad_mma_shape(mma_shape);
  }
}

#define DG_X_TILE_CASE(idx, TileType)                                          \
  case idx:                                                                    \
    return run_volume_gemm_x_tiled<TileType>(deriv_x, flux_x, D1D, escale, Nq, \
                                             Ne);

//- Shared, order-specialized x volume GEMM.  One issue point for GEMM_CUTE
//- and GEMM_FUSED so the two cannot get different tiles; `weighted` is the
//- only fused-only degree of freedom and it changes the epilogue output op,
//- not the mainloop.
extern "C" int launch_volume_gemm_x_tiled(double *deriv_x, const double *flux_x,
                                          const double *D1D,
                                          const double *escale, int weighted,
                                          int Nq, int Ne, int tile)
{
  if (Ne <= 0) {
    return 1;
  }
  if (!weighted) {
    escale = nullptr;
  }
  switch (Nq) {
  case 8:
    switch (tile) {
      DG_X_TILE_CASE(0, P7XTile0)
      DG_X_TILE_CASE(1, P7XTile1)
      DG_X_TILE_CASE(2, P7XTile2)
      DG_X_TILE_CASE(3, P7XTile3)
      DG_X_TILE_CASE(4, P7XTile4)
      DG_X_TILE_CASE(5, P7XTile5)
      DG_X_TILE_CASE(6, P7XTile6)
      DG_X_TILE_CASE(7, P7XTile7)
      DG_X_TILE_CASE(8, P7XTile8)
      DG_X_TILE_CASE(9, P7XTile9)
      DG_X_TILE_CASE(12, P7XTile12)
      DG_X_TILE_CASE(13, P7XTile13)
      DG_X_TILE_CASE(14, P7XTile14)
      DG_X_TILE_CASE(15, P7XTile15)
    default:
      break;
    }
    break;
  case 16:
    switch (tile) {
      DG_X_TILE_CASE(0, P15XTile0)
      DG_X_TILE_CASE(1, P15XTile1)
      DG_X_TILE_CASE(2, P15XTile2)
      DG_X_TILE_CASE(3, P15XTile3)
      DG_X_TILE_CASE(4, P15XTile4)
      DG_X_TILE_CASE(5, P15XTile5)
      DG_X_TILE_CASE(6, P15XTile6)
      DG_X_TILE_CASE(7, P15XTile7)
      DG_X_TILE_CASE(8, P15XTile8)
      DG_X_TILE_CASE(9, P15XTile9)
      DG_X_TILE_CASE(10, P15XTile10)
      DG_X_TILE_CASE(11, P15XTile11)
    default:
      break;
    }
    break;
  case 32:
    switch (tile) {
      DG_X_TILE_CASE(0, P31XTile0)
      DG_X_TILE_CASE(1, P31XTile1)
      DG_X_TILE_CASE(2, P31XTile2)
      DG_X_TILE_CASE(3, P31XTile3)
      DG_X_TILE_CASE(4, P31XTile4)
      DG_X_TILE_CASE(5, P31XTile5)
      DG_X_TILE_CASE(6, P31XTile6)
      DG_X_TILE_CASE(7, P31XTile7)
      DG_X_TILE_CASE(8, P31XTile8)
      DG_X_TILE_CASE(9, P31XTile9)
      DG_X_TILE_CASE(10, P31XTile10)
      DG_X_TILE_CASE(11, P31XTile11)
    default:
      break;
    }
    break;
  case 64:
    switch (tile) {
      DG_X_TILE_CASE(0, P63XTile0)
      DG_X_TILE_CASE(1, P63XTile1)
      DG_X_TILE_CASE(2, P63XTile2)
      DG_X_TILE_CASE(3, P63XTile3)
      DG_X_TILE_CASE(4, P63XTile4)
      DG_X_TILE_CASE(5, P63XTile5)
      DG_X_TILE_CASE(6, P63XTile6)
      DG_X_TILE_CASE(7, P63XTile7)
      DG_X_TILE_CASE(8, P63XTile8)
      DG_X_TILE_CASE(9, P63XTile9)
      DG_X_TILE_CASE(10, P63XTile10)
      DG_X_TILE_CASE(11, P63XTile11)
    default:
      break;
    }
    break;
  default:
    std::fprintf(stderr, "launch_volume_gemm_x_tiled: no tile set for Nq %d\n",
                 Nq);
    return 1;
  }
  std::fprintf(stderr, "launch_volume_gemm_x_tiled: bad tile %d at Nq %d\n",
               tile, Nq);
  return 1;
}

#undef DG_X_TILE_CASE

extern "C" int launch_volume_gemm_z(double *deriv_z, const double *flux_z,
                                    const double *D1D_tr, int Nq, int Ne,
                                    int mma_shape)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  if (Nq == 8 && mma_shape == 0) {
    return run_volume_gemm_z<P7MmaSet_884>(deriv_z, flux_z, D1D_tr, Nq, Ne);
  }
  if (Nq == 16 && mma_shape == 0) {
    return run_volume_gemm_z<P15MmaSet_884>(deriv_z, flux_z, D1D_tr, Nq, Ne);
  }
  switch (mma_shape) {
  case 0:
    return run_volume_gemm_z<MmaSet_884>(deriv_z, flux_z, D1D_tr, Nq, Ne);
  case 1:
    return run_volume_gemm_z<MmaSet_1684>(deriv_z, flux_z, D1D_tr, Nq, Ne);
  case 2:
    return run_volume_gemm_z<MmaSet_1688>(deriv_z, flux_z, D1D_tr, Nq, Ne);
  case 3:
    return run_volume_gemm_z<MmaSet_16816>(deriv_z, flux_z, D1D_tr, Nq, Ne);
  default:
    return bad_mma_shape(mma_shape);
  }
}

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
    const double *D1D, const double *D1D_tr, const double *escale, int Nq,
    int Ne, int mma_shape)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  switch (mma_shape) {
  case 0:
    return run_volume_gemms_xy<MmaSet_884>(deriv_x, deriv_y, flux_x,
                                           flux_y, D1D, D1D_tr,
                                           escale, Nq, Ne);
  case 1:
    return run_volume_gemms_xy<MmaSet_1684>(deriv_x, deriv_y, flux_x,
                                            flux_y, D1D, D1D_tr,
                                            escale, Nq, Ne);
  case 2:
    return run_volume_gemms_xy<MmaSet_1688>(deriv_x, deriv_y, flux_x,
                                            flux_y, D1D, D1D_tr,
                                            escale, Nq, Ne);
  case 3:
    return run_volume_gemms_xy<MmaSet_16816>(deriv_x, deriv_y, flux_x,
                                             flux_y, D1D, D1D_tr,
                                             escale, Nq, Ne);
  default:
    return bad_mma_shape(mma_shape);
  }
}

extern "C" int launch_volume_gemm_y(double *deriv_y, const double *flux_y,
                                    const double *D1D_tr, int Nq, int Ne,
                                    int mma_shape)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  if (Nq == 8 && mma_shape == 0) {
    return run_volume_gemm_y<P7MmaSet_884>(deriv_y, flux_y, D1D_tr, Nq, Ne);
  }
  if (Nq == 16 && mma_shape == 0) {
    return run_volume_gemm_y<P15MmaSet_884>(deriv_y, flux_y, D1D_tr, Nq, Ne);
  }
  if (Nq == 32) {
    switch (mma_shape) {
    case 0:
      return run_volume_gemm_y32<MmaSet_884>(deriv_y, flux_y, D1D_tr, Nq, Ne);
    case 1:
      return run_volume_gemm_y32<MmaSet_1684>(deriv_y, flux_y, D1D_tr, Nq, Ne);
    case 2:
      return run_volume_gemm_y32<MmaSet_1688>(deriv_y, flux_y, D1D_tr, Nq, Ne);
    case 3:
      return run_volume_gemm_y32<MmaSet_16816>(deriv_y, flux_y, D1D_tr, Nq, Ne);
    default:
      return bad_mma_shape(mma_shape);
  }
  }
  switch (mma_shape) {
  case 0:
    return run_volume_gemm_y<MmaSet_884>(deriv_y, flux_y, D1D_tr, Nq, Ne);
  case 1:
    return run_volume_gemm_y<MmaSet_1684>(deriv_y, flux_y, D1D_tr, Nq, Ne);
  case 2:
    return run_volume_gemm_y<MmaSet_1688>(deriv_y, flux_y, D1D_tr, Nq, Ne);
  case 3:
    return run_volume_gemm_y<MmaSet_16816>(deriv_y, flux_y, D1D_tr, Nq, Ne);
  default:
    return bad_mma_shape(mma_shape);
  }
}

extern "C" int launch_volume_gemm_y_scaleadd(
    double *deriv_y, const double *flux_y, const double *D1D_tr,
    const double *escale_y, const double *deriv_x, const double *escale_x,
    int Nq, int Ne, int mma_shape)
{
  if (Nq <= 0 || Ne <= 0 || escale_x == nullptr) {
    return 1;
  }
  switch (mma_shape) {
  case 0:
    return run_volume_gemm_y_scaleadd<MmaSet_884, true>(
        deriv_y, flux_y, D1D_tr, escale_y, deriv_x, escale_x, Nq, Ne);
  case 1:
    return run_volume_gemm_y_scaleadd<MmaSet_1684, true>(
        deriv_y, flux_y, D1D_tr, escale_y, deriv_x, escale_x, Nq, Ne);
  case 2:
    return run_volume_gemm_y_scaleadd<MmaSet_1688, true>(
        deriv_y, flux_y, D1D_tr, escale_y, deriv_x, escale_x, Nq, Ne);
  case 3:
    return run_volume_gemm_y_scaleadd<MmaSet_16816, true>(
        deriv_y, flux_y, D1D_tr, escale_y, deriv_x, escale_x, Nq, Ne);
  default:
    return bad_mma_shape(mma_shape);
  }
}

//- xy_weighted says deriv_y already holds Ex*Dx + Ey*Dy, so this kernel reads
//- two volume tensors instead of five.  GemmZWide only when Nq > 64.
extern "C" int launch_z_gemm_assembly(
    double *dqdt, const double *flux_z, const double *D1D_tr,
    const double *deriv_x, const double *deriv_y, const double *flux_bnd,
    const double *lift1d, const double *lift_zpair, const double *escale,
    int Nq, int Ne, int mma_shape, int xy_weighted)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  //- Nq = 8 keeps the five-tensor (unweighted) epilogue -- forwarding Escale
  //- into y costs far more there than z gains -- but takes the other three
  //- ingredients of the Nq > 64 package: hoisted clamps, paired 16-byte face
  //- loads and 16-byte epilogue accesses.  Together they are -3.9%; the wide
  //- epilogue and the paired loads win on their own too, while the clamp hoist
  //- only wins on top of them.  At Nq = 16, 32 and 64 every one of the three
  //- loses.  reports/p7_gemm_fused.md section 13.
  if (Nq == 8 && mma_shape == 0 && xy_weighted != 1) {
    //- xy_weighted == 2 is the partial form: CUTLASS x already applied
    //- Escale_x, cuBLAS y could not apply Escale_y, so this epilogue reads
    //- four volume tensors instead of five.  Same mainloop and same other
    //- three ingredients either way.
    if (xy_weighted == 2) {
      return run_z_gemm_assembly<P7MmaSet_884, false, true, true, true, true>(
          dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d,
          lift_zpair, escale, Nq, Ne);
    }
    return run_z_gemm_assembly<P7MmaSet_884, false, true, true, true>(
        dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d,
        lift_zpair, escale, Nq, Ne);
  }
  //- Same partial form at the two other orders whose x GEMM can carry
  //- Escale_x while y cannot.  Nq = 64 does not need it: there the y GEMM
  //- already applies Escale_x through GemmYScale.
  if (xy_weighted == 2 && mma_shape == 0 && Nq == 16) {
    return run_z_gemm_assembly<P15MmaSet_884, false, false, false, false, true>(
        dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d, lift_zpair,
        escale, Nq, Ne);
  }
  if (xy_weighted == 2 && mma_shape == 0 && Nq == 32) {
    return run_z_gemm_assembly<MmaSet_884, false, false, false, false, true>(
        dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d, lift_zpair,
        escale, Nq, Ne);
  }
  if (xy_weighted == 2) {
    std::fprintf(stderr,
                 "launch_z_gemm_assembly: no partial-weighting epilogue for "
                 "Nq %d / mma %d\n", Nq, mma_shape);
    return 1;
  }
  if (Nq == 16 && mma_shape == 0 && !xy_weighted) {
    return run_z_gemm_assembly<P15MmaSet_884, false>(
        dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d,
        lift_zpair, escale, Nq, Ne);
  }
  if (!xy_weighted) {
    switch (mma_shape) {
    case 0:
      return run_z_gemm_assembly<MmaSet_884, false>(dqdt, flux_z, D1D_tr, deriv_x,
                                                    deriv_y, flux_bnd, lift1d,
                                                    lift_zpair, escale, Nq, Ne);
    case 1:
      return run_z_gemm_assembly<MmaSet_1684, false>(dqdt, flux_z, D1D_tr, deriv_x,
                                                     deriv_y, flux_bnd, lift1d,
                                                     lift_zpair, escale, Nq, Ne);
    case 2:
      return run_z_gemm_assembly<MmaSet_1688, false>(dqdt, flux_z, D1D_tr, deriv_x,
                                                     deriv_y, flux_bnd, lift1d,
                                                     lift_zpair, escale, Nq, Ne);
    case 3:
      return run_z_gemm_assembly<MmaSet_16816, false>(dqdt, flux_z, D1D_tr, deriv_x,
                                                      deriv_y, flux_bnd, lift1d,
                                                      lift_zpair, escale, Nq, Ne);
    default:
      return bad_mma_shape(mma_shape);
    }
  }
  if (Nq == 64) {
    switch (mma_shape) {
    case 0:
      return run_z_gemm_assembly<MmaSet_884, true, false>(
          dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d, lift_zpair,
          escale, Nq, Ne);
    case 1:
      return run_z_gemm_assembly<MmaSet_1684, true, false>(
          dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d, lift_zpair,
          escale, Nq, Ne);
    case 2:
      return run_z_gemm_assembly<MmaSet_1688, true, false>(
          dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d, lift_zpair,
          escale, Nq, Ne);
    case 3:
      return run_z_gemm_assembly<MmaSet_16816, true, false>(
          dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d, lift_zpair,
          escale, Nq, Ne);
    default:
      return bad_mma_shape(mma_shape);
    }
  }
  switch (mma_shape) {
  case 0:
    return run_z_gemm_assembly<MmaSet_884, true>(dqdt, flux_z, D1D_tr, deriv_x,
                                           deriv_y, flux_bnd, lift1d,
                                           lift_zpair, escale, Nq, Ne);
  case 1:
    return run_z_gemm_assembly<MmaSet_1684, true>(dqdt, flux_z, D1D_tr, deriv_x,
                                            deriv_y, flux_bnd, lift1d,
                                            lift_zpair, escale, Nq, Ne);
  case 2:
    return run_z_gemm_assembly<MmaSet_1688, true>(dqdt, flux_z, D1D_tr, deriv_x,
                                            deriv_y, flux_bnd, lift1d,
                                            lift_zpair, escale, Nq, Ne);
  case 3:
    return run_z_gemm_assembly<MmaSet_16816, true>(dqdt, flux_z, D1D_tr, deriv_x,
                                             deriv_y, flux_bnd, lift1d,
                                             lift_zpair, escale, Nq, Ne);
  default:
    return bad_mma_shape(mma_shape);
  }
}


//- GEMM_FUSED with the assembly epilogue moved from z onto y.  The mainloop is
//- Set::GemmY (Nq = 16) / Set::GemmY32 (Nq = 32), i.e. exactly the y mainloop
//- GEMM_CUTE uses, so the two paths keep the same tiles and the same library
//- assignment; only the epilogue differs, which AGENTS.md does not share.
//- x_weighted == 2 means the x GEMM already folded Escale_x into deriv_x.
extern "C" int launch_y_gemm_assembly(
    double *dqdt, const double *flux_y, const double *D1D_tr,
    const double *deriv_x, const double *deriv_z, const double *flux_bnd,
    const double *lift1d, const double *escale, int Nq, int Ne, int mma_shape,
    int x_weighted)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  if (mma_shape != 0) {
    return bad_mma_shape(mma_shape);
  }
  if (Nq == 16) {
    if (x_weighted == 2) {
      return run_y_gemm_assembly<P15MmaSet_884::GemmY, true>(
          dqdt, flux_y, D1D_tr, deriv_x, deriv_z, flux_bnd, lift1d, escale, Nq, Ne);
    }
    return run_y_gemm_assembly<P15MmaSet_884::GemmY, false>(
        dqdt, flux_y, D1D_tr, deriv_x, deriv_z, flux_bnd, lift1d, escale, Nq, Ne);
  }
  if (Nq == 32) {
    if (x_weighted == 2) {
      return run_y_gemm_assembly<MmaSet_884::GemmY32, true>(
          dqdt, flux_y, D1D_tr, deriv_x, deriv_z, flux_bnd, lift1d, escale, Nq, Ne);
    }
    return run_y_gemm_assembly<MmaSet_884::GemmY32, false>(
        dqdt, flux_y, D1D_tr, deriv_x, deriv_z, flux_bnd, lift1d, escale, Nq, Ne);
  }
  std::fprintf(stderr, "launch_y_gemm_assembly: unsupported Nq %d\n", Nq);
  return 1;
}
