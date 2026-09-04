#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

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
#include "cutlass_x_gemm_assembly.h"

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

//- The four MMA instruction shapes the namelist entry CutlassMmaShape selects,
//- as (InstructionShape, ArchTag, threadblock TileK).  On sm_90 each of the
//- three 16x8xK forms is one SASS instruction; on sm_100 ptxas expands them
//- into 2 / 4 / 8 DMMA.8x8x4, which is why the tree's default is 8x8x4 and why
//- the shape only pays on H100 (reports/h100_report.md section 4,
//- reports/sm90_mma_shape_survey.md section 6).
//-
//- kTileK is not free: MmaBase asserts WarpShape::kK / InstShape::kK >= 2 and
//- even, so 16x8x16 needs TileK = 32.  Shapes 0-2 share TileK = 16 and can
//- therefore be dropped into any tile the 8x8x4 path already uses; shape 3
//- cannot, and stays confined to the generic VolumeGemmSet as before.
template <int kShape> struct MmaShapeSel;
template <> struct MmaShapeSel<0> {
  using Inst = GS<8, 8, 4>;
  using Arch = cutlass::arch::Sm80;
  static constexpr int kTileK = 16;
};
template <> struct MmaShapeSel<1> {
  using Inst = GS<16, 8, 4>;
  using Arch = cutlass::arch::Sm90;
  static constexpr int kTileK = 16;
};
template <> struct MmaShapeSel<2> {
  using Inst = GS<16, 8, 8>;
  using Arch = cutlass::arch::Sm90;
  static constexpr int kTileK = 16;
};
template <> struct MmaShapeSel<3> {
  using Inst = GS<16, 8, 16>;
  using Arch = cutlass::arch::Sm90;
  static constexpr int kTileK = 32;
};

//- Shapes 0-2 are the ones the order-specialized tiles and sets below are
//- instantiated for.  kMaxTiledShape names that limit in one place so the
//- launchers can say why shape 3 is refused there.
constexpr int kMaxTiledShape = 2;

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

  //- Escale-forwarding twins of the PLAIN y / y32 / z GEMMs above: same tile,
  //- same warp partition, same stage count, only the epilogue output op
  //- differs.  They exist for the x carrier, where y and z are both plain
  //- GEMMs and can therefore each absorb their own Escale factor.  (GemmYScale
  //- above is NOT usable for that: it is four stages deep, which would make
  //- GEMM_FUSED and GEMM_CUTE differ in the mainloop.)
  using GemmYIsoScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 64, TileK>, GS<32, 32, TileK>, InstShape,
      PointwiseScaleV<1>, BatchedSwizzle, 3>;
  using GemmY32Scale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<32, 32, TileK>, GS<16, 16, TileK>, InstShape,
      PointwiseScaleV<1>, BatchedSwizzle, 3>;
  using GemmZScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<64, 32, TileK>, GS<32, 32, TileK>, InstShape,
      PointwiseScaleV<1>, BatchedSwizzle, 4>;

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

//- The generic set indexed by shape, so a launcher can write one dispatch
//- instead of a four-way switch per call site.
template <int kShape> using GenericMmaSet =
    VolumeGemmSet<typename MmaShapeSel<kShape>::Inst,
                  typename MmaShapeSel<kShape>::Arch,
                  MmaShapeSel<kShape>::kTileK>;
static_assert(std::is_same<GenericMmaSet<0>, MmaSet_884>::value, "shape 0 drifted");
static_assert(std::is_same<GenericMmaSet<1>, MmaSet_1684>::value, "shape 1 drifted");
static_assert(std::is_same<GenericMmaSet<2>, MmaSet_1688>::value, "shape 2 drifted");
static_assert(std::is_same<GenericMmaSet<3>, MmaSet_16816>::value, "shape 3 drifted");

// Nq=8 makes the generic tiles mostly predicated. CUTLASS's FP64 TensorOp
// multistage iterator needs at least two warps; the measured winners are
// 32x64 for the 8x8 y batches and 16x32 for the 64x8 z/assembly GEMM.
template <class InstShape, class ArchTag, int TileK>
struct VolumeGemmSetP7 : VolumeGemmSet<InstShape, ArchTag, TileK> {
  using GemmY = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<32, 64, TileK>, GS<32, 32, TileK>,
      InstShape, EpilogueOp, BatchedSwizzle, 3>;
  using GemmYScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<32, 64, TileK>, GS<32, 32, TileK>,
      InstShape, PointwiseScaleV<1>, BatchedSwizzle, 3>;
  using GemmZ = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<16, 32, TileK>, GS<16, 16, TileK>,
      InstShape, EpilogueOp, BatchedSwizzle, 3>;
  //- Same tile with 16-byte epilogue accesses.  Only the low-order study of
  //- reports/p7_gemm_fused.md section 13 instantiates it; the mainloop is the
  //- one GemmZ has, so GEMM_CUTE and GEMM_FUSED still share it.
  using GemmZWide = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<16, 32, TileK>, GS<16, 16, TileK>,
      InstShape, EpilogueOp2, BatchedSwizzle, 3>;
  //- Escale-forwarding twins of this set's plain y and z, for the x carrier.
  using GemmYIsoScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<32, 64, TileK>, GS<32, 32, TileK>,
      InstShape, PointwiseScaleV<1>, BatchedSwizzle, 3>;
  using GemmZScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<16, 32, TileK>, GS<16, 16, TileK>,
      InstShape, PointwiseScaleV<1>, BatchedSwizzle, 3>;
};

using P7MmaSet_884 = VolumeGemmSetP7<GS<8, 8, 4>, cutlass::arch::Sm80, 16>;
//- The same set at MMA instruction shape kShape, for shapes 0-2 (they share
//- TileK = 16).  Both GEMM_CUTE and GEMM_FUSED select through this alias, so
//- a non-default shape keeps the order's tiles instead of falling back to the
//- generic 64x128 / 64x64 / 64x32 set.
template <int kShape> using P7MmaSet =
    VolumeGemmSetP7<typename MmaShapeSel<kShape>::Inst,
                    typename MmaShapeSel<kShape>::Arch,
                    MmaShapeSel<kShape>::kTileK>;
static_assert(std::is_same<P7MmaSet<0>, P7MmaSet_884>::value,
              "Nq=8 default volume GEMM set drifted");

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
//- InstShape / ArchTag default to the 8x8x4 pair, so every tile alias below
//- and every existing use site keeps exactly the type it had.  A non-default
//- MMA shape reaches this tile only through XTileS<> further down, which is
//- what lets GEMM_CUTE and GEMM_FUSED run the SAME tile at a non-default
//- shape instead of one of them silently falling back to the generic set.
template <int TbM, int TbN, int WM, int WN, int TileK = 16, int Stages = 3,
          class InstShape = GS<8, 8, 4>, class ArchTag = cutlass::arch::Sm80>
struct XTile {
  using GemmX = cutlass::gemm::device::Gemm<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<TbM, TbN, TileK>, GS<WM, WN, TileK>,
      InstShape, EpilogueOp, Swizzle, Stages>;
  using GemmXScale = cutlass::gemm::device::Gemm<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<TbM, TbN, TileK>, GS<WM, WN, TileK>,
      InstShape, PointwiseScaleV<1>, Swizzle, Stages>;
  //- Same threadblock tile, warp partition, MMA shape, swizzle and stage
  //- count with 16-byte epilogue accesses (kElementsPerAccess = 2).  Only the
  //- carrier instantiates it, and only the epilogue differs, so GEMM_CUTE and
  //- GEMM_FUSED still share one mainloop.  Same construction as GemmZWide.
  using GemmXWide = cutlass::gemm::device::Gemm<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<TbM, TbN, TileK>, GS<WM, WN, TileK>,
      InstShape, EpilogueOp2, Swizzle, Stages>;
};

//- The same tile at MMA instruction shape kShape.  TileK stays the tile's own
//- (16 everywhere the order-specialized lists use it), which is legal for
//- shapes 0-2; shape 3 would need 32 and is not offered here.
template <int kShape, int TbM, int TbN, int WM, int WN, int TileK = 16,
          int Stages = 3>
using XTileS = XTile<TbM, TbN, WM, WN, TileK, Stages,
                     typename MmaShapeSel<kShape>::Inst,
                     typename MmaShapeSel<kShape>::Arch>;

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

//- Nq > 64 x tiles.  Above 64 the transposed problem's N is Nq = 128 or 256,
//- so the generic GemmX (64x128) is already unpredicated at Nq = 128 and
//- covers N in two tiles at Nq = 256 -- which is why no order-specialized x
//- tile ever existed here.  Candidate 0 IS that generic tile, bit for bit, so
//- SCALE_DG_XTILE=0 and the untiled launch_volume_gemm_x must measure the
//- same; the rest vary the long-axis extent, the warp partition, TileK and
//- the pipeline depth.  One list serves both orders: the tile shape depends
//- on Nq only through N = Nq, and TbN = 128 is the widest that fits.
using PHXTile0 = XTile<64, 128, 32, 64, 16, 3>;
using PHXTile1 = XTile<128, 128, 64, 64, 16, 3>;
using PHXTile2 = XTile<128, 128, 32, 64, 16, 3>;
using PHXTile3 = XTile<64, 128, 32, 32, 16, 3>;
using PHXTile4 = XTile<32, 128, 16, 64, 16, 3>;
using PHXTile5 = XTile<64, 64, 32, 32, 16, 3>;
using PHXTile6 = XTile<64, 128, 32, 64, 16, 4>;
using PHXTile7 = XTile<64, 128, 32, 64, 32, 3>;
using PHXTile8 = XTile<256, 128, 64, 64, 16, 3>;
using PHXTile9 = XTile<64, 128, 64, 64, 16, 3>;
using PHXTile10 = XTile<128, 64, 64, 32, 16, 3>;
using PHXTile11 = XTile<64, 128, 16, 64, 16, 3>;
//- Carrier-motivated candidates.  The epilogue's per-thread fragment is
//- TbM*TbN/threads elements, and the x carrier holds FIVE of them (the
//- accumulator plus Dy, Dz, Ex and the lift, or seven with Ey and Ez): tiles
//- 0-11 all give 32 or 64 doubles per thread at Nq >= 128 and spill.  These
//- four are the 16-doubles-per-thread shapes, i.e. the same fragment size the
//- Nq > 64 z carrier's 64x32 tile has, reached four different ways.
using PHXTile12 = XTile<64, 32, 32, 16, 16, 3>;
using PHXTile13 = XTile<64, 64, 16, 32, 16, 3>;
using PHXTile15 = XTile<128, 32, 64, 16, 16, 3>;
//- Tile 14 (64x128 with a 16x32 warp, 512 threads) is NOT usable: the plain x
//- GEMM built from it faults with an illegal address at Nq = 128, so it stays
//- out of both dispatch tables.
//- Refinements around tile 5, the carrier winner of the first pass.
using PHXTile16 = XTile<32, 64, 16, 32, 16, 3>;
using PHXTile17 = XTile<128, 64, 32, 32, 16, 3>;
using PHXTile18 = XTile<64, 64, 32, 32, 16, 4>;
using PHXTile19 = XTile<64, 64, 32, 16, 16, 3>;

//- The ADOPTED x tile of each order, expressed once per MMA instruction shape.
//- The x GEMM is the one volume GEMM whose tile is order-specialized at every
//- order up to 256, and until now those tiles hard-coded GS<8,8,4>: selecting
//- CutlassMmaShape moved the y and z GEMMs and left x on 8x8x4 in BOTH
//- GEMM_CUTE and GEMM_FUSED, and the fused x carrier (Nq = 8, 16, 32) refused
//- the shape outright.  These aliases close that.  Only the adopted tile of
//- each order is instantiated at a non-default shape: the ablation menus above
//- exist to choose a tile at 8x8x4, and instantiating all of them four times
//- over would multiply the build for measurements nobody asked for.  The
//- static_asserts below are the guarantee that shape 0 is the very type the
//- named tile alias already was, so the default path cannot drift.
template <int S> using XAdopted8 = XTileS<S, 64, 16, 16, 16, 16, 3>;
template <int S> using XAdopted16 = XTileS<S, 64, 16, 16, 16, 16, 3>;
template <int S> using XAdopted32 = XTileS<S, 32, 32, 16, 16, 16, 3>;
template <int S> using XAdopted64 = XTileS<S, 128, 64, 64, 32, 16, 3>;
template <int S> using XAdoptedHi = XTileS<S, 64, 128, 32, 64, 16, 3>;

static_assert(std::is_same<XAdopted8<0>, P7XTile9>::value,
              "Nq=8 adopted x tile drifted from P7XTile9");
static_assert(std::is_same<XAdopted16<0>, P15XTile5>::value,
              "Nq=16 adopted x tile drifted from P15XTile5");
static_assert(std::is_same<XAdopted32<0>, P31XTile7>::value,
              "Nq=32 adopted x tile drifted from P31XTile7");
static_assert(std::is_same<XAdopted64<0>, P63XTile3>::value,
              "Nq=64 adopted x tile drifted from P63XTile3");
static_assert(std::is_same<XAdoptedHi<0>, PHXTile0>::value,
              "Nq>64 adopted x tile drifted from PHXTile0");

//- Order-specialized PLAIN z volume GEMM tiles.  The z GEMM is
//- (m = Nq*Nq, n = Nq, k = Nq) per element with a column-major C, so
//- GemmBatched runs the transposed problem: the threadblock's M dimension
//- covers Nq and its N dimension covers the long Nq*Nq axis.  That is the
//- mirror image of the x GEMM, where M was long and N was Nq.  Only the
//- plain z uses these -- both GEMM_CUTE and GEMM_FUSED reach it through
//- launch_volume_gemm_z, so the two paths keep one mainloop -- and the
//- carrier z (run_z_gemm_assembly) keeps the set's own GemmZ.
template <int TbM, int TbN, int WM, int WN, int TileK = 16, int Stages = 3>
struct ZTile {
  using GemmZ = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, cutlass::arch::Sm80, GS<TbM, TbN, TileK>, GS<WM, WN, TileK>,
      GS<8, 8, 4>, EpilogueOp, BatchedSwizzle, Stages>;
};

//- Nq = 16 candidates.  Tile 0 is the shape VolumeGemmSetP15::GemmZ already
//- uses, so SCALE_DG_ZTILE=0 and the default must measure the same.
using P15ZTile0 = ZTile<16, 64, 16, 32, 16, 3>;
using P15ZTile1 = ZTile<16, 128, 16, 32, 16, 3>;
using P15ZTile2 = ZTile<16, 32, 16, 16, 16, 3>;
using P15ZTile3 = ZTile<16, 256, 16, 64, 16, 3>;
using P15ZTile4 = ZTile<16, 64, 16, 16, 16, 3>;
using P15ZTile5 = ZTile<16, 128, 16, 64, 16, 3>;
using P15ZTile6 = ZTile<16, 64, 16, 32, 32, 3>;
using P15ZTile7 = ZTile<16, 64, 16, 32, 16, 4>;
using P15ZTile8 = ZTile<16, 64, 16, 32, 16, 5>;
using P15ZTile9 = ZTile<32, 64, 32, 32, 16, 3>;
using P15ZTile10 = ZTile<16, 128, 16, 64, 16, 4>;
using P15ZTile11 = ZTile<16, 256, 16, 32, 16, 3>;

//- Generic (Nq >= 32) plain-z candidates.  ABLATION ONLY, for the CTA
//- fixed-cost study of reports/gemm_assignment_and_carrier.md section 10.10:
//- the adopted plan keeps z_tile = -1 at every order, so production dispatch
//- never reaches these and GEMM_CUTE / GEMM_FUSED keep one shared mainloop.
//- Tile 0 IS the shape VolumeGemmSet::GemmZ already uses, so SCALE_DG_ZTILE=0
//- and the default must measure the same.  1..4 vary ONLY the multistage
//- depth and 8 only the mainloop K step: both move the mainloop PROLOGUE
//- (pipeline fill) and leave the CTA count, hence the last-wave TAIL,
//- untouched.  5 keeps the depth and halves the CTA count (TbN 32 -> 64),
//- which moves the tail and leaves the prologue untouched; 7 keeps the CTA
//- count and doubles the warps per CTA.  6 widens TbM instead, which halves
//- the CTA count only where Nq > 64 and otherwise predicates half the tile
//- away -- it is a predication control, not a tail control.  Stages = 2 is
//- not instantiable: CUTLASS routes it to the two-stage pipelined MmaCore,
//- which is ambiguous for FP64 TensorOp.
using ZG0 = ZTile<64, 32, 32, 32, 16, 4>;
using ZG1 = ZTile<64, 32, 32, 32, 16, 3>;
using ZG2 = ZTile<64, 32, 32, 32, 16, 5>;
using ZG3 = ZTile<64, 32, 32, 32, 16, 6>;
using ZG4 = ZTile<64, 32, 32, 32, 16, 8>;
using ZG5 = ZTile<64, 64, 32, 32, 16, 4>;
using ZG6 = ZTile<128, 32, 32, 32, 16, 4>;
using ZG7 = ZTile<64, 32, 16, 32, 16, 4>;
using ZG8 = ZTile<64, 32, 32, 32, 32, 4>;

//- Nq=16 leaves the generic tiles as predicated as Nq=8 does: the y GEMM is
//- 16x16x16 per batch against a 64x64 tile, and the z / z-assembly GEMM is
//- 256x16x16 against a 64x32 tile whose N half is predicated away.  Same
//- reasoning as VolumeGemmSetP7; the shapes below are the measured winners
//- (reports/p15_gap_study.md).  Both GEMM_CUTE and GEMM_FUSED take them, so
//- the two paths keep sharing one volume-GEMM mainloop.
template <class InstShape, class ArchTag, int TileK>
struct VolumeGemmSetP15 : VolumeGemmSet<InstShape, ArchTag, TileK> {
  using GemmY = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<16, 32, TileK>,
      GS<16, 16, TileK>, InstShape, EpilogueOp, BatchedSwizzle, 3>;
  using GemmYScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<16, 32, TileK>,
      GS<16, 16, TileK>, InstShape, PointwiseScaleV<1>, BatchedSwizzle, 3>;
  using GemmZ = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<16, 64, TileK>,
      GS<16, 32, TileK>, InstShape, EpilogueOp, BatchedSwizzle, 3>;
  using GemmZWide = GemmZ;
  //- Escale-forwarding twins of this set's plain y and z, for the x carrier.
  using GemmYIsoScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<16, 32, TileK>,
      GS<16, 16, TileK>, InstShape, PointwiseScaleV<1>, BatchedSwizzle, 3>;
  using GemmZScale = cutlass::gemm::device::GemmBatched<
      double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
      TensorOp, ArchTag, GS<16, 64, TileK>,
      GS<16, 32, TileK>, InstShape, PointwiseScaleV<1>, BatchedSwizzle, 3>;
};

using P15MmaSet_884 = VolumeGemmSetP15<GS<8, 8, 4>, cutlass::arch::Sm80, 16>;
template <int kShape> using P15MmaSet =
    VolumeGemmSetP15<typename MmaShapeSel<kShape>::Inst,
                     typename MmaShapeSel<kShape>::Arch,
                     MmaShapeSel<kShape>::kTileK>;
static_assert(std::is_same<P15MmaSet<0>, P15MmaSet_884>::value,
              "Nq=16 default volume GEMM set drifted");

//- OUT-OF-ROLE ABLATION (SCALE_DG_ZS3, default 0): the same volume-GEMM set
//- with every z mainloop one multistage stage shallower.  Restage<> rebuilds a
//- device::GemmBatched from the typedefs the original exposes, so the tile,
//- warp partition, MMA shape, epilogue output op and swizzle are carried over
//- verbatim and only Stages changes.  Both z mainloops are restaged together:
//- GemmZ is the plain z that GEMM_CUTE (and a non-z carrier of GEMM_FUSED)
//- runs, GemmZWide is the one the z carrier's assembly kernel runs.  Changing
//- only one of them would make GEMM_CUTE and GEMM_FUSED differ in the volume
//- mainloop, which AGENTS.md forbids; changing both is the "shared mainloop
//- change" the study of gemm_assignment_and_carrier.md section 10.10.8 asked
//- for.  Default dispatch never reaches these (the knob defaults to 0).
template <class G, int Stages>
using Restage = cutlass::gemm::device::GemmBatched<
    double, ColumnMajor, double, ColumnMajor, double, ColumnMajor, double,
    TensorOp, typename G::ArchTag, typename G::ThreadblockShape,
    typename G::WarpShape, typename G::InstructionShape,
    typename G::EpilogueOutputOp, typename G::ThreadblockSwizzle, Stages>;

template <class Set>
struct ZStage3Set : Set {
  using GemmZ = Restage<typename Set::GemmZ, 3>;
  using GemmZWide = Restage<typename Set::GemmZWide, 3>;
};

//- ADOPTED (job 79164, gemm_assignment_and_carrier.md section 12): the shared
//- z mainloop runs three stages deep at Nq = 64, 128 and 256 and four
//- everywhere else.  Both z mainloops move together and both GEMM_CUTE and
//- GEMM_FUSED reach them through launch_volume_gemm_z /
//- launch_z_gemm_assembly, so the two paths keep one tile at every order --
//- which is what AGENTS.md requires of an order-specialized mainloop.  A
//- negative zs3 argument means "take this table"; 0 and 1 force the depth,
//- which is how the A/B was measured.
inline bool z_stage3_default(int Nq)
{
  return Nq == 64 || Nq == 128 || Nq == 256;
}

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

//- The order-specialized tiles and sets are instantiated for shapes 0-2 only
//- (shape 3 needs TileK = 32, which is a different tile, not a different
//- instruction).  Say which limit was hit rather than "unsupported".
int bad_tiled_mma_shape(int mma_shape, int Nq)
{
  std::fprintf(stderr,
               "cutlass volume gemm: mma_shape %d has no order-specialized "
               "tile at Nq %d (0-2 are instantiated; 16x8x16 would need "
               "TileK = 32, i.e. a different tile)\n",
               mma_shape, Nq);
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
  //- The Nq >= 512 branch below is GEMM_FUSED's x; GEMM_CUTE's x takes
  //- run_volume_gemm_x, which names Set::GemmY.  AGENTS.md requires the two
  //- to share one mainloop, so assert it rather than trusting the comments:
  //- GemmYScaleShallow differs from GemmY in the epilogue output op alone
  //- (both 64x64 / warp 32x32 / TileK / 3 stages).  Both launchers also go
  //- through the pad-8 RepadEpilogue (p767_gap_study.md 14.1), so the x
  //- epilogue divergence that existed at Nq = 128 / 256 has no twin here.
  static_assert(
      cutlass::platform::is_same<
          typename Set::GemmY::GemmKernel::Mma,
          typename Set::GemmYScaleShallow::GemmKernel::Mma>::value,
      "Nq>=512: GEMM_CUTE's x/y mainloop must equal GEMM_FUSED's");
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

//- Plain y / z volume GEMMs with their own Escale factor folded into the
//- stock epilogue.  Same tile, warps and stage count as run_volume_gemm_y /
//- _y32 / _z; only the output op differs, so GEMM_CUTE and GEMM_FUSED keep one
//- mainloop.  Only the x carrier can use these: with y or z carrying the
//- assembly epilogue, the carrier's own Escale multiplies its own accumulator
//- and cannot be forwarded anywhere.
template <class Set>
int run_volume_gemm_y_scaled(double *deriv_y, const double *flux_y,
                             const double *D1D_tr, const double *escale_y,
                             int Nq, int Ne)
{
  const long long stride_plane = Nq * Nq;
  return run_gemm_batched_nn_scaled<typename Set::GemmYIsoScale>(
      Nq, Nq, Nq, flux_y, Nq, stride_plane, D1D_tr, Nq, 0, escale_y, Nq,
      stride_plane, deriv_y, Nq, stride_plane, Nq * Ne);
}

template <class Set>
int run_volume_gemm_y32_scaled(double *deriv_y, const double *flux_y,
                               const double *D1D_tr, const double *escale_y,
                               int Nq, int Ne)
{
  const long long stride_plane = Nq * Nq;
  return run_gemm_batched_nn_scaled<typename Set::GemmY32Scale>(
      Nq, Nq, Nq, flux_y, Nq, stride_plane, D1D_tr, Nq, 0, escale_y, Nq,
      stride_plane, deriv_y, Nq, stride_plane, Nq * Ne);
}

template <class Set>
int run_volume_gemm_z_scaled(double *deriv_z, const double *flux_z,
                             const double *D1D_tr, const double *escale_z,
                             int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const long long stride_vol = static_cast<long long>(nq2) * Nq;
  return run_gemm_batched_nn_scaled<typename Set::GemmZScale>(
      nq2, Nq, Nq, flux_z, nq2, stride_vol, D1D_tr, Nq, 0, escale_z, nq2,
      stride_vol, deriv_z, nq2, stride_vol, Ne);
}

//- z volume GEMM whose epilogue folds the y term in: deriv_yz = Escale_z * acc
//- + deriv_y, where deriv_y already holds Escale_y * Dy (the x carrier's
//- Escale_y forward, run_volume_gemm_y_scaled).  This is the mirror of
//- run_volume_gemm_y_scaleadd, which the Nq > 64 z carrier uses to fold
//- deriv_x into y, and it exists for the same reason: the carrier's epilogue
//- then reads ONE volume tensor instead of two.  Mainloop, tile, warps,
//- swizzle and stage count are Set::GemmZScale's, i.e. the ones GEMM_CUTE
//- runs for the plain z; only the epilogue differs, so the two paths keep one
//- mainloop.  Fused-only, and only with an x carrier: with z carrying there is
//- nothing to fold into.
template <class Set>
int run_volume_gemm_z_scaleadd(double *deriv_yz, const double *flux_z,
                               const double *D1D_tr, const double *escale_z,
                               const double *deriv_y, int Nq, int Ne)
{
  const int nq2 = Nq * Nq;
  const long long stride_vol = static_cast<long long>(nq2) * Nq;

  using GemmZ = typename Set::GemmZScale;
  using Kernel = GemmBatchedScaleAdd<typename GemmZ::GemmKernel::Mma,
                                     RepadEpilogue<typename GemmZ::GemmKernel::Epilogue, 8>,
                                     BatchedSwizzle, false>;
  static_assert(cutlass::platform::is_same<
                    typename Set::GemmZ::GemmKernel::Mma,
                    typename GemmZ::GemmKernel::Mma>::value,
                "the folded z must share the plain z mainloop");

  cutlass::TensorRef<double const, ColumnMajor> ref_A(flux_z, nq2);
  cutlass::TensorRef<double const, ColumnMajor> ref_B(D1D_tr, Nq);
  cutlass::TensorRef<double, ColumnMajor> ref_D(deriv_yz, nq2);

  typename GemmZ::Arguments gemm_args({nq2, Nq, Nq}, ref_A, stride_vol, ref_B, 0,
                                      ref_D, stride_vol, ref_D, stride_vol,
                                      typename GemmZ::EpilogueOutputOp::Params(), Ne);
  cutlass::Status can = GemmZ::can_implement(gemm_args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("z-scaleadd can_implement", can);
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
      uargs.stride_C, uargs.ref_D, uargs.stride_D, uargs.epilogue, uargs.batch_count);
  params.ptr_scale = escale_z;
  params.ptr_add = deriv_y;
  params.ptr_mul = nullptr;
  params.stride_scale = stride_vol;
  params.stride_add = stride_vol;
  params.stride_mul = 0;

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

//- The same fusion package on the x GEMM.  y and z run first as plain
//- unweighted (or Escale-forwarded) GEMMs into deriv_y and a scratch deriv_z,
//- and this kernel -- whose mainloop is the order-specialized x tile GEMM_CUTE
//- runs -- applies Escale_x/y/z, adds the six-face lift and writes dqdt.
//- See cutlass_x_gemm_assembly.h for the geometry.
//- ABLATION for the Nq = 64 x carrier.  cutlass::Kernel carries no launch
//- bounds, so ptxas allocates up to 254 registers and the 128x64 tile fits
//- two CTAs per SM.  This wrapper is the identical kernel with an explicit
//- occupancy floor, to price what capping the register budget buys or costs.
template <typename Operator, int MinCTA>
__global__ __launch_bounds__(Operator::kThreadCount, MinCTA)
void DgKernelLaunchBounds(typename Operator::Params params)
{
  extern __shared__ int SharedStorageBase[];
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);
  Operator op;
  op(params, *shared_storage);
}

template <class Tile, bool kYWeighted, bool kZWeighted, bool kAffine = false,
          bool kPaired = false, bool kWide = false, int kMinCTA = 0,
          bool kFold = false>
int run_x_gemm_assembly(double *dqdt, const double *flux_x, const double *D1D,
                        const double *deriv_y, const double *deriv_z,
                        const double *flux_bnd, const double *lift1d,
                        const double *lift_pair, const double *escale, int Nq,
                        int Ne, bool fast_index)
{
  //- kWide swaps only the epilogue output op (16-byte accesses); the
  //- mainloop tile, warps, swizzle and stage count are the shared ones.
  using GemmX = typename cutlass::platform::conditional<
      kWide, typename Tile::GemmXWide, typename Tile::GemmX>::type;
  //- Machine check of the rule GEMM_CUTE and GEMM_FUSED share: the wide
  //- variant differs in the epilogue output op only, so its mainloop type --
  //- tile, warps, MMA, swizzle, stages, iterators -- must be bit-identical to
  //- the one GEMM_CUTE runs for this order's plain x GEMM.
  static_assert(cutlass::platform::is_same<
                    typename Tile::GemmX::GemmKernel::Mma,
                    typename Tile::GemmXWide::GemmKernel::Mma>::value,
                "GemmXWide must share GemmX's mainloop");
  using XEpilogue = RepadEpilogue<typename GemmX::GemmKernel::Epilogue, 8>;
  using Kernel = GemmDqdtAssemblyX<typename GemmX::GemmKernel::Mma, XEpilogue,
                                   Swizzle, kYWeighted, kZWeighted, kAffine,
                                   kPaired, kFold>;

  const int nq2 = Nq * Nq;
  const int Np = nq2 * Nq;
  const std::int64_t npoint = std::int64_t{Np} * Ne;
  const int n = nq2 * Ne;

  typename GemmX::Arguments args({Nq, n, Nq}, {D1D, Nq}, {flux_x, Nq},
                                 {dqdt, Nq}, {dqdt, Nq},
                                 typename GemmX::EpilogueOutputOp::Params());
  const cutlass::Status can = GemmX::can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    return cutlass_error("x-assembly can_implement", can);
  }

  //- The column-major-C device operator runs the transposed problem, so build
  //- Params from its underlying arguments rather than from ours.
  auto uargs = GemmX::to_underlying_arguments(args);
  Swizzle swizzle;
  cutlass::gemm::GemmCoord grid_shape = swizzle.get_tiled_shape(
      uargs.problem_size,
      {GemmX::ThreadblockShape::kM, GemmX::ThreadblockShape::kN,
       GemmX::ThreadblockShape::kK},
      1);

  typename Kernel::Params params;
  params.gemm = typename Kernel::BaseKernel::Params(
      uargs.problem_size, grid_shape, uargs.ref_A.non_const_ref(),
      uargs.ref_B.non_const_ref(), uargs.ref_C.non_const_ref(), uargs.ref_D,
      uargs.epilogue, nullptr);
  params.ptr_dy = deriv_y;
  params.ptr_dz = deriv_z;
  params.ptr_flux_bnd = flux_bnd;
  params.ptr_lift1d = lift1d;
  params.ptr_lift_pair = lift_pair;
  params.Nq = Nq;
  params.nq_log2 = -1;
  if (fast_index) {
    for (int b = 0; b < 16; ++b) {
      if ((1 << b) == Nq) {
        params.nq_log2 = b;
        break;
      }
    }
  }
  params.ptr_ex = escale;
  params.ptr_ey = escale + npoint;
  params.ptr_ez = escale + 2 * npoint;

  dim3 grid = swizzle.get_grid_shape(grid_shape);
  dim3 block(Kernel::kThreadCount, 1, 1);
  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr =
        kMinCTA > 0
            ? cudaFuncSetAttribute(
                  DgKernelLaunchBounds<Kernel, kMinCTA <= 0 ? 1 : kMinCTA>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)
            : cudaFuncSetAttribute(
                  cutlass::Kernel<Kernel>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (attr != cudaSuccess) {
      return 1;
    }
  }

  cutlass::arch::synclog_setup();
  if (kMinCTA > 0) {
    DgKernelLaunchBounds<Kernel, kMinCTA <= 0 ? 1 : kMinCTA>
        <<<grid, block, smem_size, dg_cuda_stream>>>(params);
  } else {
    cutlass::Kernel<Kernel><<<grid, block, smem_size, dg_cuda_stream>>>(params);
  }
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
                                          int Nq, int Ne, int tile,
                                          int mma_shape)
{
  if (Ne <= 0) {
    return 1;
  }
  if (!weighted) {
    escale = nullptr;
  }
  //- A non-default MMA instruction shape runs the ADOPTED tile of the order,
  //- not the generic one.  Until this existed the x GEMM was the one volume
  //- GEMM CutlassMmaShape could not move at Nq <= 256: its tile hard-coded
  //- 8x8x4, so selecting 16x8x4 changed y and z and silently left x behind in
  //- BOTH GEMM_CUTE and GEMM_FUSED.  Only the adopted tile is instantiated per
  //- shape; the ablation menus below are 8x8x4 measurements and stay that way.
  if (mma_shape != 0) {
    if (mma_shape > kMaxTiledShape) {
      return bad_tiled_mma_shape(mma_shape, Nq);
    }
#define DG_X_ADOPTED(TileTpl)                                                  \
    if (mma_shape == 1) {                                                      \
      return run_volume_gemm_x_tiled<TileTpl<1> >(deriv_x, flux_x, D1D,        \
                                                  escale, Nq, Ne);             \
    }                                                                          \
    return run_volume_gemm_x_tiled<TileTpl<2> >(deriv_x, flux_x, D1D, escale,  \
                                                Nq, Ne);

    if (Nq == 8 && tile == 9) { DG_X_ADOPTED(XAdopted8) }
    if (Nq == 16 && tile == 5) { DG_X_ADOPTED(XAdopted16) }
    if (Nq == 32 && tile == 7) { DG_X_ADOPTED(XAdopted32) }
    if (Nq == 64 && tile == 3) { DG_X_ADOPTED(XAdopted64) }
    if ((Nq == 128 || Nq == 256) && tile == 0) { DG_X_ADOPTED(XAdoptedHi) }
#undef DG_X_ADOPTED
    std::fprintf(stderr,
                 "launch_volume_gemm_x_tiled: mma_shape %d is instantiated on "
                 "the adopted tile only (Nq %d, tile %d)\n",
                 mma_shape, Nq, tile);
    return 1;
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
  case 128:
  case 256:
    switch (tile) {
      DG_X_TILE_CASE(0, PHXTile0)
      DG_X_TILE_CASE(1, PHXTile1)
      DG_X_TILE_CASE(2, PHXTile2)
      DG_X_TILE_CASE(3, PHXTile3)
      DG_X_TILE_CASE(4, PHXTile4)
      DG_X_TILE_CASE(5, PHXTile5)
      DG_X_TILE_CASE(6, PHXTile6)
      DG_X_TILE_CASE(7, PHXTile7)
      DG_X_TILE_CASE(8, PHXTile8)
      DG_X_TILE_CASE(9, PHXTile9)
      DG_X_TILE_CASE(10, PHXTile10)
      DG_X_TILE_CASE(11, PHXTile11)
      DG_X_TILE_CASE(12, PHXTile12)
      DG_X_TILE_CASE(13, PHXTile13)
      DG_X_TILE_CASE(15, PHXTile15)
      DG_X_TILE_CASE(16, PHXTile16)
      DG_X_TILE_CASE(17, PHXTile17)
      DG_X_TILE_CASE(18, PHXTile18)
      DG_X_TILE_CASE(19, PHXTile19)
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

//- tile >= 0 selects an explicit plain-z tile from the order's candidate list
//- (SCALE_DG_ZTILE); -1 takes the order's adopted set.  Shared by GEMM_CUTE
//- and GEMM_FUSED, which both reach the plain z here.
#define DG_Z_TILE_CASE(n)                                                      \
  case n:                                                                      \
    return run_volume_gemm_z<P15ZTile##n>(deriv_z, flux_z, D1D_tr, Nq, Ne);

//- Same, for the Nq >= 32 ablation list above.
#define DG_ZG_TILE_CASE(n)                                                     \
  case n:                                                                      \
    return run_volume_gemm_z<ZG##n>(deriv_z, flux_z, D1D_tr, Nq, Ne);

extern "C" int launch_volume_gemm_z(double *deriv_z, const double *flux_z,
                                    const double *D1D_tr, int Nq, int Ne,
                                    int mma_shape, int tile, int zs3)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  //- OUT-OF-ROLE ABLATION (SCALE_DG_ZS3): the shipped mainloop with one fewer
  //- multistage stage.  It goes with the same change on the carrier z in
  //- launch_z_gemm_assembly, so GEMM_CUTE and GEMM_FUSED move together.
  if (zs3 < 0) {
    zs3 = z_stage3_default(Nq) ? 1 : 0;
  }
  if (zs3 != 0 && mma_shape == 0 && tile < 0) {
    if (Nq == 8) {
      return run_volume_gemm_z<ZStage3Set<P7MmaSet_884> >(deriv_z, flux_z,
                                                          D1D_tr, Nq, Ne);
    }
    if (Nq == 16) {
      return run_volume_gemm_z<ZStage3Set<P15MmaSet_884> >(deriv_z, flux_z,
                                                           D1D_tr, Nq, Ne);
    }
    return run_volume_gemm_z<ZStage3Set<MmaSet_884> >(deriv_z, flux_z, D1D_tr,
                                                      Nq, Ne);
  }
  if (Nq == 16 && mma_shape == 0 && tile >= 0) {
    switch (tile) {
    DG_Z_TILE_CASE(0)
    DG_Z_TILE_CASE(1)
    DG_Z_TILE_CASE(2)
    DG_Z_TILE_CASE(3)
    DG_Z_TILE_CASE(4)
    DG_Z_TILE_CASE(5)
    DG_Z_TILE_CASE(6)
    DG_Z_TILE_CASE(7)
    DG_Z_TILE_CASE(8)
    DG_Z_TILE_CASE(9)
    DG_Z_TILE_CASE(10)
    DG_Z_TILE_CASE(11)
    default:
      std::fprintf(stderr, "launch_volume_gemm_z: no z tile %d at Nq %d\n",
                   tile, Nq);
      return 1;
    }
  }
  if (Nq >= 32 && mma_shape == 0 && tile >= 0) {
    switch (tile) {
    DG_ZG_TILE_CASE(0)
    DG_ZG_TILE_CASE(1)
    DG_ZG_TILE_CASE(2)
    DG_ZG_TILE_CASE(3)
    DG_ZG_TILE_CASE(4)
    DG_ZG_TILE_CASE(5)
    DG_ZG_TILE_CASE(6)
    DG_ZG_TILE_CASE(7)
    DG_ZG_TILE_CASE(8)
    default:
      std::fprintf(stderr, "launch_volume_gemm_z: no z tile %d at Nq %d\n",
                   tile, Nq);
      return 1;
    }
  }
  //- Same for the plain z: the order's own tile at the selected shape.
  if (Nq == 8) {
    switch (mma_shape) {
    case 0: return run_volume_gemm_z<P7MmaSet<0> >(deriv_z, flux_z, D1D_tr, Nq, Ne);
    case 1: return run_volume_gemm_z<P7MmaSet<1> >(deriv_z, flux_z, D1D_tr, Nq, Ne);
    case 2: return run_volume_gemm_z<P7MmaSet<2> >(deriv_z, flux_z, D1D_tr, Nq, Ne);
    default: return bad_tiled_mma_shape(mma_shape, Nq);
    }
  }
  if (Nq == 16) {
    switch (mma_shape) {
    case 0: return run_volume_gemm_z<P15MmaSet<0> >(deriv_z, flux_z, D1D_tr, Nq, Ne);
    case 1: return run_volume_gemm_z<P15MmaSet<1> >(deriv_z, flux_z, D1D_tr, Nq, Ne);
    case 2: return run_volume_gemm_z<P15MmaSet<2> >(deriv_z, flux_z, D1D_tr, Nq, Ne);
    default: return bad_tiled_mma_shape(mma_shape, Nq);
    }
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

#undef DG_Z_TILE_CASE

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
  //- The order-specialized y sets, now at the selected instruction shape too.
  //- Before, a non-default shape fell through to the generic 64x64 tile here,
  //- which changed the TILE as well as the instruction and so was not the
  //- one-axis measurement the knob is for.
  if (Nq == 8) {
    switch (mma_shape) {
    case 0: return run_volume_gemm_y<P7MmaSet<0> >(deriv_y, flux_y, D1D_tr, Nq, Ne);
    case 1: return run_volume_gemm_y<P7MmaSet<1> >(deriv_y, flux_y, D1D_tr, Nq, Ne);
    case 2: return run_volume_gemm_y<P7MmaSet<2> >(deriv_y, flux_y, D1D_tr, Nq, Ne);
    default: return bad_tiled_mma_shape(mma_shape, Nq);
    }
  }
  if (Nq == 16) {
    switch (mma_shape) {
    case 0: return run_volume_gemm_y<P15MmaSet<0> >(deriv_y, flux_y, D1D_tr, Nq, Ne);
    case 1: return run_volume_gemm_y<P15MmaSet<1> >(deriv_y, flux_y, D1D_tr, Nq, Ne);
    case 2: return run_volume_gemm_y<P15MmaSet<2> >(deriv_y, flux_y, D1D_tr, Nq, Ne);
    default: return bad_tiled_mma_shape(mma_shape, Nq);
    }
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
//- pkg >= 0 overrides the three epilogue ingredients of the Nq > 64 package on
//- the z carrier -- bit 0 clamp aggregation (kAffine), bit 1 16-byte lift
//- (kPaired), bit 2 16-byte epilogue (GemmZWide) -- so they can be measured at
//- Nq = 16 and 64, where the adopted forms are (off,off,off) and (on,on,off).
//- All three are epilogue-side; the mainloop tile is the set's GemmZ either
//- way (GemmZWide is built from the same tile, warps, swizzle and stages).
#define DG_Z_ASM_PKG(Set, WEIGHTED)                                            \
  switch (pkg & 7) {                                                           \
  case 0: return DG_Z_ASM_CALL(Set, WEIGHTED, false, false, false);            \
  case 1: return DG_Z_ASM_CALL(Set, WEIGHTED, false, true, false);             \
  case 2: return DG_Z_ASM_CALL(Set, WEIGHTED, false, false, true);             \
  case 3: return DG_Z_ASM_CALL(Set, WEIGHTED, false, true, true);              \
  case 4: return DG_Z_ASM_CALL(Set, WEIGHTED, true, false, false);             \
  case 5: return DG_Z_ASM_CALL(Set, WEIGHTED, true, true, false);              \
  case 6: return DG_Z_ASM_CALL(Set, WEIGHTED, true, false, true);              \
  default: return DG_Z_ASM_CALL(Set, WEIGHTED, true, true, true);              \
  }

#define DG_Z_ASM_CALL(Set, WEIGHTED, WD, AF, PR)                               \
  run_z_gemm_assembly<Set, WEIGHTED, WD, AF, PR>(                              \
      dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d, lift_zpair,    \
      escale, Nq, Ne)

extern "C" int launch_z_gemm_assembly(
    double *dqdt, const double *flux_z, const double *D1D_tr,
    const double *deriv_x, const double *deriv_y, const double *flux_bnd,
    const double *lift1d, const double *lift_zpair, const double *escale,
    int Nq, int Ne, int mma_shape, int xy_weighted, int pkg, int zs3)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  //- OUT-OF-ROLE ABLATION (SCALE_DG_ZS3): the carrier z with a three-deep
  //- pipeline.  Only the two configurations production dispatch actually
  //- reaches at Nq >= 64 are instantiated -- the Nq = 64 package (16-byte
  //- epilogue + clamp aggregation, no 16-byte lift) and the Nq > 64 default
  //- (all three on) -- so the knob does not multiply the build.
  const bool zs3_forced = (zs3 >= 0);
  if (zs3 < 0) {
    zs3 = z_stage3_default(Nq) ? 1 : 0;
  }
  if (zs3 != 0 && mma_shape == 0 && xy_weighted == 1) {
    if (Nq == 64 && pkg == 5) {
      return run_z_gemm_assembly<ZStage3Set<MmaSet_884>, true, true, true,
                                 false>(dqdt, flux_z, D1D_tr, deriv_x, deriv_y,
                                        flux_bnd, lift1d, lift_zpair, escale,
                                        Nq, Ne);
    }
    if (Nq > 64 && pkg < 0) {
      return run_z_gemm_assembly<ZStage3Set<MmaSet_884>, true>(
          dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d, lift_zpair,
          escale, Nq, Ne);
    }
  }
  //- Only the two configurations production dispatch reaches are built with
  //- the shallower pipeline.  Anything else -- an ablation carrier, an
  //- SCALE_DG_ZPKG override, the SCALE_DG_XYW ablation -- falls through to the
  //- four-stage form, which is correct and is what those knobs measured
  //- against anyway.  Say so when the depth was asked for explicitly.
  if (zs3 != 0 && zs3_forced) {
    std::fprintf(stderr,
                 "launch_z_gemm_assembly: no three-stage carrier at Nq %d "
                 "(xy_weighted %d, pkg %d, mma %d); using four\n", Nq,
                 xy_weighted, pkg, mma_shape);
  }
  //- OUT-OF-ROLE ABLATION (SCALE_DG_XYW=0 at Nq > 64): the five-tensor
  //- epilogue with the SAME three package ingredients the weighted Nq > 64
  //- form uses (16-byte epilogue, clamp aggregation, 16-byte paired face
  //- loads).  Keeping the package fixed is what makes this a one-axis
  //- measurement of the Escale/deriv_x fold alone; it is also required for
  //- correctness, because the driver still has elembnd_flux_kernel write the
  //- interleaved face layout at Nq >= 64.
  if (xy_weighted == 0 && pkg < 0 && mma_shape == 0 && Nq > 64) {
    return run_z_gemm_assembly<MmaSet_884, false, true, true, true>(
        dqdt, flux_z, D1D_tr, deriv_x, deriv_y, flux_bnd, lift1d, lift_zpair,
        escale, Nq, Ne);
  }
  if (pkg >= 0 && mma_shape == 0 && Nq == 64 && xy_weighted == 1) {
    DG_Z_ASM_PKG(MmaSet_884, true)
  }
  //- The same carrier at a non-default MMA shape.  Only the ADOPTED package of
  //- Nq = 64 (zpkg 5: 16-byte epilogue and clamp aggregation, no 16-byte lift)
  //- is instantiated; the eight-way package sweep stays an 8x8x4 measurement.
  //- The mainloop is the generic set's z, which GEMM_CUTE runs at the same
  //- shape through launch_volume_gemm_z, so the two paths still share it.
  if (pkg == 5 && mma_shape != 0 && Nq == 64 && xy_weighted == 1) {
    switch (mma_shape) {
    case 1: return DG_Z_ASM_CALL(MmaSet_1684, true, true, true, false);
    case 2: return DG_Z_ASM_CALL(MmaSet_1688, true, true, true, false);
    case 3: return DG_Z_ASM_CALL(MmaSet_16816, true, true, true, false);
    default: return bad_mma_shape(mma_shape);
    }
  }
  if (pkg >= 0 && mma_shape == 0 && Nq == 16 && xy_weighted == 0) {
    DG_Z_ASM_PKG(P15MmaSet_884, false)
  }
  if (pkg >= 0) {
    std::fprintf(stderr,
                 "launch_z_gemm_assembly: no package override at Nq %d "
                 "(xy_weighted %d)\n", Nq, xy_weighted);
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
#undef DG_Z_ASM_PKG
#undef DG_Z_ASM_CALL

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
  if (Nq == 8) {
    //- The Nq = 8 y mainloop is the 32x64 tile of VolumeGemmSetP7, i.e. what
    //- GEMM_CUTE runs for y at this order once y is on CUTLASS at all.  Making
    //- y the carrier forces that library choice on BOTH paths; it is not a
    //- carrier-only tile.
    if (x_weighted == 2) {
      return run_y_gemm_assembly<P7MmaSet_884::GemmY, true>(
          dqdt, flux_y, D1D_tr, deriv_x, deriv_z, flux_bnd, lift1d, escale, Nq, Ne);
    }
    return run_y_gemm_assembly<P7MmaSet_884::GemmY, false>(
        dqdt, flux_y, D1D_tr, deriv_x, deriv_z, flux_bnd, lift1d, escale, Nq, Ne);
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
  if (Nq == 128 || Nq == 256) {
    //- Nq > 64.  The y mainloop is MmaSet_884::GemmY (64x64, warp 32x32, three
    //- stages), which is what GEMM_CUTE runs for y at these orders, so the two
    //- paths keep one tile.  The batched problem is (128,128) or (256,256) per
    //- batch, so a 64x64 tile is unpredicated -- the fit that decided the
    //- Nq = 32 y carrier.  What it cannot have is the Nq > 64 epilogue package:
    //- cutlass_y_gemm_assembly.h has neither the 16-byte epilogue nor the
    //- clamp aggregation, and that package is worth 5-6% to the x carrier here.
    if (x_weighted == 2) {
      return run_y_gemm_assembly<MmaSet_884::GemmY, true>(
          dqdt, flux_y, D1D_tr, deriv_x, deriv_z, flux_bnd, lift1d, escale, Nq, Ne);
    }
    return run_y_gemm_assembly<MmaSet_884::GemmY, false>(
        dqdt, flux_y, D1D_tr, deriv_x, deriv_z, flux_bnd, lift1d, escale, Nq, Ne);
  }
  std::fprintf(stderr, "launch_y_gemm_assembly: unsupported Nq %d\n", Nq);
  return 1;
}

//- Plain y volume GEMM with Escale_y folded into its own epilogue.  Only the
//- x carrier reaches this: with y carrying the assembly epilogue Escale_y
//- multiplies its own accumulator and cannot be forwarded, and with z carrying
//- it the forwarding target is the z epilogue's kWeighted branch instead.
extern "C" int launch_volume_gemm_y_scaled(double *deriv_y, const double *flux_y,
                                           const double *D1D_tr,
                                           const double *escale_y, int Nq,
                                           int Ne, int mma_shape)
{
  if (Nq <= 0 || Ne <= 0 || escale_y == nullptr) {
    return 1;
  }
  if (mma_shape != 0) {
    return bad_mma_shape(mma_shape);
  }
  if (Nq == 8) {
    return run_volume_gemm_y_scaled<P7MmaSet_884>(deriv_y, flux_y, D1D_tr,
                                                  escale_y, Nq, Ne);
  }
  if (Nq == 16) {
    return run_volume_gemm_y_scaled<P15MmaSet_884>(deriv_y, flux_y, D1D_tr,
                                                   escale_y, Nq, Ne);
  }
  if (Nq == 32) {
    return run_volume_gemm_y32_scaled<MmaSet_884>(deriv_y, flux_y, D1D_tr,
                                                  escale_y, Nq, Ne);
  }
  return run_volume_gemm_y_scaled<MmaSet_884>(deriv_y, flux_y, D1D_tr, escale_y,
                                              Nq, Ne);
}

//- Same for the plain z volume GEMM.
//- Folded z for the x carrier: deriv_yz = Escale_z * D(flux_z) + deriv_y, with
//- deriv_y already weighted by Escale_y.  Same mainloop as the plain z above.
extern "C" int launch_volume_gemm_z_scaleadd(
    double *deriv_yz, const double *flux_z, const double *D1D_tr,
    const double *escale_z, const double *deriv_y, int Nq, int Ne,
    int mma_shape)
{
  if (Nq <= 0 || Ne <= 0 || deriv_y == nullptr) {
    return 1;
  }
  switch (mma_shape) {
  case 0:
    return run_volume_gemm_z_scaleadd<MmaSet_884>(deriv_yz, flux_z, D1D_tr,
                                                  escale_z, deriv_y, Nq, Ne);
  case 1:
    return run_volume_gemm_z_scaleadd<MmaSet_1684>(deriv_yz, flux_z, D1D_tr,
                                                   escale_z, deriv_y, Nq, Ne);
  case 2:
    return run_volume_gemm_z_scaleadd<MmaSet_1688>(deriv_yz, flux_z, D1D_tr,
                                                   escale_z, deriv_y, Nq, Ne);
  case 3:
    return run_volume_gemm_z_scaleadd<MmaSet_16816>(deriv_yz, flux_z, D1D_tr,
                                                    escale_z, deriv_y, Nq, Ne);
  default:
    return bad_mma_shape(mma_shape);
  }
}

extern "C" int launch_volume_gemm_z_scaled(double *deriv_z, const double *flux_z,
                                           const double *D1D_tr,
                                           const double *escale_z, int Nq,
                                           int Ne, int mma_shape)
{
  if (Nq <= 0 || Ne <= 0 || escale_z == nullptr) {
    return 1;
  }
  if (mma_shape != 0) {
    return bad_mma_shape(mma_shape);
  }
  if (Nq == 8) {
    return run_volume_gemm_z_scaled<P7MmaSet_884>(deriv_z, flux_z, D1D_tr,
                                                  escale_z, Nq, Ne);
  }
  if (Nq == 16) {
    return run_volume_gemm_z_scaled<P15MmaSet_884>(deriv_z, flux_z, D1D_tr,
                                                   escale_z, Nq, Ne);
  }
  return run_volume_gemm_z_scaled<MmaSet_884>(deriv_z, flux_z, D1D_tr, escale_z,
                                              Nq, Ne);
}

//- GEMM_FUSED with the assembly epilogue moved onto x.  The mainloop is the
//- order's adopted CUTLASS x tile, i.e. exactly the x mainloop GEMM_CUTE runs,
//- so the two paths keep the same tiles and the same library assignment; only
//- the epilogue differs, which AGENTS.md does not share.
//- Only the ADOPTED tile of each order is instantiated: the tile is shared
//- with GEMM_CUTE by rule, so a carrier on a different tile could never be
//- adopted, and four weight combinations times twelve candidate tiles times
//- four orders is not a compile the sweep needs.
//- weight_mode bit 0: Escale_y already folded into deriv_y by the y GEMM.
//- weight_mode bit 1: Escale_z already folded into deriv_z by the z GEMM.
//- weight_mode bit 2: ABLATION.  Force the general integer-division row-index
//- split instead of the shift/mask one, to price that arithmetic.
//- pkg: the three Nq > 64 epilogue ingredients, which the z carrier bundles
//- as kAffine / kPaired / GemmZWide.  bit 0 = clamp aggregation (kAffine),
//- bit 1 = 16-byte lift, i.e. the interleaved face layout and the packed
//- lift-coefficient table (kPaired), bit 2 = 16-byte epilogue accesses
//- (GemmXWide).  All three are epilogue-side; the mainloop tile, warps,
//- swizzle and stage count are the ones GEMM_CUTE runs either way.
#define DG_X_ASM_PKG(TileType, WY, WZ)                                         \
  switch (pkg & 7) {                                                           \
  case 0: return DG_X_ASM_CALL(TileType, WY, WZ, false, false, false);         \
  case 1: return DG_X_ASM_CALL(TileType, WY, WZ, true, false, false);          \
  case 2: return DG_X_ASM_CALL(TileType, WY, WZ, false, true, false);          \
  case 3: return DG_X_ASM_CALL(TileType, WY, WZ, true, true, false);           \
  case 4: return DG_X_ASM_CALL(TileType, WY, WZ, false, false, true);          \
  case 5: return DG_X_ASM_CALL(TileType, WY, WZ, true, false, true);           \
  case 6: return DG_X_ASM_CALL(TileType, WY, WZ, false, true, true);           \
  default: return DG_X_ASM_CALL(TileType, WY, WZ, true, true, true);           \
  }

#define DG_X_ASM_CALL(TileType, WY, WZ, AF, PR, WD)                            \
  run_x_gemm_assembly<TileType, WY, WZ, AF, PR, WD>(                           \
      dqdt, flux_x, D1D, deriv_y, deriv_z, flux_bnd, lift1d, lift_pair,        \
      escale, Nq, Ne, (weight_mode & 4) == 0)

//- ABLATION, Nq = 64 only: the same carrier under an explicit occupancy floor.
//- pkg bits 3-4 carry it (1 -> 2 CTAs/SM, 2 -> 3 CTAs/SM); it is instantiated
//- only for the two package forms the Nq = 64 sweep found best, both with
//- Escale_y and Escale_z forwarded.
#define DG_X_ASM_LB(AF, MIN)                                                   \
  run_x_gemm_assembly<P63XTile3, true, true, AF, false, false, MIN>(           \
      dqdt, flux_x, D1D, deriv_y, deriv_z, flux_bnd, lift1d, lift_pair,        \
      escale, Nq, Ne, (weight_mode & 4) == 0)

//- ABLATION / ceiling measurement, Nq = 64 tile 3 only: the same carrier with
//- the y term folded into the z GEMM's epilogue (kFold), so the epilogue reads
//- one volume tensor instead of two.  Both Escale factors are forwarded by
//- construction.  weight_mode bit 3 selects it; pkg bit 0 is clamp
//- aggregation and pkg bits 3-4 the occupancy floor, the same encoding the
//- unfolded ceiling used.
#define DG_X_ASM_FOLD(AF, MIN)                                                 \
  run_x_gemm_assembly<P63XTile3, true, true, AF, false, false, MIN, true>(     \
      dqdt, flux_x, D1D, deriv_y, deriv_z, flux_bnd, lift1d, lift_pair,        \
      escale, Nq, Ne, (weight_mode & 4) == 0)

//- Sweep-only carrier instantiation: one package (both Escale forwarded,
//- clamp aggregation, 16-byte epilogue, no 16-byte lift), which is the form
//- the Nq > 64 package sweep on tile 0 found best.  A tile that wins here is
//- promoted to the full DG_X_ASM_CASE before it can be adopted.
#define DG_X_ASM_ONE(TileType)                                                 \
  return DG_X_ASM_CALL(TileType, true, true, true, false, true);

#define DG_X_ASM_CASE(TileType)                                                \
  switch (weight_mode & 3) {                                                   \
  case 0:                                                                      \
    DG_X_ASM_PKG(TileType, false, false)                                       \
  case 1:                                                                      \
    DG_X_ASM_PKG(TileType, true, false)                                        \
  case 2:                                                                      \
    DG_X_ASM_PKG(TileType, false, true)                                        \
  default:                                                                     \
    DG_X_ASM_PKG(TileType, true, true)                                         \
  }

extern "C" int launch_x_gemm_assembly(
    double *dqdt, const double *flux_x, const double *D1D,
    const double *deriv_y, const double *deriv_z, const double *flux_bnd,
    const double *lift1d, const double *lift_pair, const double *escale,
    int Nq, int Ne, int mma_shape, int tile, int weight_mode, int pkg)
{
  if (Nq <= 0 || Ne <= 0) {
    return 1;
  }
  //- The fused x carrier at a non-default MMA shape.  Instantiated for the
  //- ADOPTED (tile, weight_mode, package) of the three orders that actually
  //- carry on x -- Nq = 8 (tile 9, xpkg 5), 16 (tile 5, xpkg 4) and 32
  //- (tile 7, xpkg 4), all with no Escale forwarded -- and for nothing else.
  //- DG_X_ASM_CASE is a 4 x 8 sweep over weight_mode and package; building it
  //- per shape as well would triple this file for combinations no order uses.
  //- The tile is XAdopted*<shape>, the same type launch_volume_gemm_x_tiled
  //- hands GEMM_CUTE at that shape, so the two paths keep one x mainloop --
  //- which is the AGENTS.md requirement that made this instantiation
  //- necessary rather than optional.
  if (mma_shape != 0) {
    if (mma_shape > kMaxTiledShape) {
      return bad_tiled_mma_shape(mma_shape, Nq);
    }
    if ((weight_mode & 11) == 0) {
      if (Nq == 8 && tile == 9 && pkg == 5) {
        if (mma_shape == 1) {
          return DG_X_ASM_CALL(XAdopted8<1>, false, false, true, false, true);
        }
        return DG_X_ASM_CALL(XAdopted8<2>, false, false, true, false, true);
      }
      if (Nq == 16 && tile == 5 && pkg == 4) {
        if (mma_shape == 1) {
          return DG_X_ASM_CALL(XAdopted16<1>, false, false, false, false, true);
        }
        return DG_X_ASM_CALL(XAdopted16<2>, false, false, false, false, true);
      }
      if (Nq == 32 && tile == 7 && pkg == 4) {
        if (mma_shape == 1) {
          return DG_X_ASM_CALL(XAdopted32<1>, false, false, false, false, true);
        }
        return DG_X_ASM_CALL(XAdopted32<2>, false, false, false, false, true);
      }
    }
    std::fprintf(stderr,
                 "launch_x_gemm_assembly: mma_shape %d is instantiated only on "
                 "the adopted carrier of Nq = 8, 16 and 32 (got Nq %d tile %d "
                 "weight_mode %d pkg %d)\n",
                 mma_shape, Nq, tile, weight_mode, pkg);
    return 1;
  }
  if ((weight_mode & 8) != 0 && !(Nq == 64 && tile == 3)) {
    std::fprintf(stderr,
                 "launch_x_gemm_assembly: the folded y term is instantiated "
                 "only at Nq = 64 on tile 3 (got Nq %d tile %d)\n",
                 Nq, tile);
    return 1;
  }
  if (Nq == 64 && tile == 3 && (weight_mode & 8) != 0) {
    switch (((pkg >> 3) & 3) * 2 + (pkg & 1)) {
    case 0: return DG_X_ASM_FOLD(false, 0);
    case 1: return DG_X_ASM_FOLD(true, 0);
    case 2: return DG_X_ASM_FOLD(false, 2);
    case 3: return DG_X_ASM_FOLD(true, 2);
    case 4: return DG_X_ASM_FOLD(false, 3);
    case 5: return DG_X_ASM_FOLD(true, 3);
    default:
      std::fprintf(stderr, "launch_x_gemm_assembly: bad folded pkg %d\n", pkg);
      return 1;
    }
  }
  if (Nq == 64 && tile == 3 && (pkg >> 3) != 0) {
    switch (((pkg >> 3) & 3) * 2 + (pkg & 1)) {
    case 2: return DG_X_ASM_LB(false, 2);
    case 3: return DG_X_ASM_LB(true, 2);
    case 4: return DG_X_ASM_LB(false, 3);
    case 5: return DG_X_ASM_LB(true, 3);
    default:
      std::fprintf(stderr, "launch_x_gemm_assembly: bad launch-bound pkg %d\n",
                   pkg);
      return 1;
    }
  }
  switch (Nq) {
  case 8:
    if (tile != 9) break;
    DG_X_ASM_CASE(P7XTile9)
  case 16:
    if (tile != 5) break;
    DG_X_ASM_CASE(P15XTile5)
  case 32:
    if (tile != 7) break;
    DG_X_ASM_CASE(P31XTile7)
  case 64:
    if (tile != 3) break;
    DG_X_ASM_CASE(P63XTile3)
  case 128:
  case 256:
    //- Nq > 64: two candidates are instantiated, the generic-equal tile 0 and
    //- the fastest alternative the plain-x sweep found.  The carrier's tile is
    //- shared with GEMM_CUTE by rule, so a carrier on any other tile could
    //- never be adopted.
    if (tile == 0) {
      DG_X_ASM_CASE(PHXTile0)
    }
    if (tile == 5) {
      DG_X_ASM_CASE(PHXTile5)
    }
    if (tile == 18) {
      DG_X_ASM_CASE(PHXTile18)
    }
    switch (tile) {
    case 3: DG_X_ASM_ONE(PHXTile3)
    case 11: DG_X_ASM_ONE(PHXTile11)
    case 12: DG_X_ASM_ONE(PHXTile12)
    case 13: DG_X_ASM_ONE(PHXTile13)
    case 15: DG_X_ASM_ONE(PHXTile15)
    case 16: DG_X_ASM_ONE(PHXTile16)
    case 17: DG_X_ASM_ONE(PHXTile17)
    case 19: DG_X_ASM_ONE(PHXTile19)
    default: break;
    }
    break;
  default:
    std::fprintf(stderr, "launch_x_gemm_assembly: unsupported Nq %d\n", Nq);
    return 1;
  }
  std::fprintf(stderr,
               "launch_x_gemm_assembly: tile %d at Nq %d is not instantiated "
               "(only the adopted tile is)\n",
               tile, Nq);
  return 1;
}

#undef DG_X_ASM_CASE
#undef DG_X_ASM_ONE
#undef DG_X_ASM_PKG
#undef DG_X_ASM_CALL
#undef DG_X_ASM_LB
#undef DG_X_ASM_FOLD
