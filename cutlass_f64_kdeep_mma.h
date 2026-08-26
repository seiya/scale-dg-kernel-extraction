#pragma once

//- Makes the SM90 FP64 MMA shapes with K > 4 (m16n8k8, m16n8k16) usable from
//- the CUTLASS 2.x TensorOp pipeline.
//-
//- The 64-bit warp tile iterators in
//- cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h are built around a
//- four-deep K group: their access index is `c + s * Iterations::kContiguous`,
//- and the crosswise variant applies its 64-bit swizzle correction once per
//- fragment from a single `k_group_idx_`. Both are right when the instruction
//- is K=4 deep (8x8x4, 16x8x4) and wrong otherwise: with K=8 the two K groups
//- end up interleaved across the M/N atoms, while mma.sync.m16n8k8 wants each
//- atom's K to be contiguous, and only one of the two groups gets the swizzle
//- parity it needs.
//-
//- Rather than reimplement the swizzle, the iterator here drives the stock
//- K=4 iterator kGroups times and concatenates the results per atom. That is
//- exactly the operand order of the wide instruction: for f64, m16n8k8's A is
//- two m16n8k4 A operands stacked along K, and likewise for B and for
//- m16n8k16. The accumulator layout is the same for all three shapes, so the
//- epilogue is untouched.
//-
//- On sm_100 this buys nothing -- ptxas expands m16n8k4/8/16 into 2/4/8
//- DMMA.8x8x4 -- but on sm_90 each is a single instruction. See
//- reports/sm90_mma_shape_survey.md.

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"

namespace dg_f64_kdeep {

using cutlass::gemm::Operand;

/// Warp tile iterator for an instruction that is more than four elements deep
/// in K. Wraps the stock K=4 iterator and reorders its fragments so that each
/// MMA atom owns a contiguous run of K.
template <typename Shape_, Operand kOperand_, typename Element_,
          typename Layout_, typename InstructionShape_, int OpDelta_,
          int Threads, int PartitionsK_>
class KDeepMultiplicandTileIterator {
public:
  using Shape = Shape_;
  static Operand const kOperand = kOperand_;
  using Element = Element_;
  using Layout = Layout_;
  using InstructionShape = InstructionShape_;
  static int const kOpDelta = OpDelta_;
  static int const kThreads = Threads;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "KDeepMultiplicandTileIterator handles A and B only.");

  //- K is the column extent for A and the row extent for B.
  static int const kInstructionK =
      (kOperand == Operand::kA) ? InstructionShape::kColumn : InstructionShape::kRow;
  static int const kGroups = kInstructionK / 4;

  static_assert(kInstructionK % 4 == 0 && kGroups >= 2,
                "This iterator is only for K-deep instructions (K = 8, 16).");

  /// The same instruction, cut down to one four-deep K group.
  using GroupInstructionShape = typename cutlass::platform::conditional<
      kOperand == Operand::kA,
      cutlass::MatrixShape<InstructionShape::kRow, 4>,
      cutlass::MatrixShape<4, InstructionShape::kColumn>>::type;

  using Base = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
      Shape, kOperand, Element, Layout, GroupInstructionShape, kOpDelta,
      kThreads, PartitionsK_>;

  using TensorRef = typename Base::TensorRef;
  using TensorCoord = typename Base::TensorCoord;
  using Index = typename Base::Index;
  using LongIndex = typename Base::LongIndex;

  using GroupFragment = typename Base::Fragment;
  using Fragment =
      cutlass::Array<Element, GroupFragment::kElements * kGroups>;

  /// Number of MMA atoms the warp tile spans in the non-K dimension.
  static int const kAtoms =
      (kOperand == Operand::kA) ? Shape::kRow / InstructionShape::kRow
                                : Shape::kColumn / InstructionShape::kColumn;
  /// Registers one atom takes from one K group.
  static int const kRegsPerAtom = GroupFragment::kElements / kAtoms;

  static_assert(kRegsPerAtom * kAtoms == GroupFragment::kElements,
                "The K=4 fragment must divide evenly among the atoms.");

private:
  Base iterator_;

  /// Tile offsets arrive in units of this iterator's instruction, which is
  /// kGroups times deeper in K than the one underneath.
  CUTLASS_HOST_DEVICE
  static TensorCoord scale_k(TensorCoord const &tile_offset)
  {
    return (kOperand == Operand::kA)
               ? TensorCoord{tile_offset.row(), tile_offset.column() * kGroups}
               : TensorCoord{tile_offset.row() * kGroups, tile_offset.column()};
  }

public:
  CUTLASS_HOST_DEVICE
  KDeepMultiplicandTileIterator() {}

  CUTLASS_HOST_DEVICE
  KDeepMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_(ref, lane_id) {}

  CUTLASS_HOST_DEVICE
  KDeepMultiplicandTileIterator &add_pointer_offset(LongIndex offset)
  {
    iterator_.add_pointer_offset(offset);
    return *this;
  }

  CUTLASS_HOST_DEVICE
  KDeepMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset)
  {
    iterator_.add_tile_offset(scale_k(tile_offset));
    return *this;
  }

  CUTLASS_HOST_DEVICE
  KDeepMultiplicandTileIterator &operator+=(TensorCoord const &tile_offset)
  {
    return add_tile_offset(tile_offset);
  }

  CUTLASS_HOST_DEVICE
  KDeepMultiplicandTileIterator &operator++()
  {
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < kGroups; ++g) {
      ++iterator_;
    }
    return *this;
  }

  CUTLASS_HOST_DEVICE
  void set_kgroup_index(int k_group)
  {
    iterator_.set_kgroup_index(k_group * kGroups);
  }

  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const
  {
    Base walker = iterator_;
    Element *dst = reinterpret_cast<Element *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < kGroups; ++g) {
      GroupFragment group;
      walker.load(group);
      Element const *src = reinterpret_cast<Element const *>(&group);

      CUTLASS_PRAGMA_UNROLL
      for (int a = 0; a < kAtoms; ++a) {
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < kRegsPerAtom; ++r) {
          dst[(a * kGroups + g) * kRegsPerAtom + r] = src[a * kRegsPerAtom + r];
        }
      }

      ++walker;
    }
  }
};

/// The stock warp-level TensorOp with the two multiplicand iterators replaced.
/// Everything else -- transform(), the mma loop, the accumulator iterator --
/// is inherited unchanged, and the fragment types are the same Array<double,N>
/// as the base's, only filled in a different order.
template <typename Shape_, typename ElementA_, typename LayoutA_,
          typename ElementB_, typename LayoutB_, typename ElementC_,
          typename LayoutC_, typename Policy_, int PartitionsK_ = 1,
          bool AccumulatorsInRowMajor = false>
class MmaTensorOpKDeep
    : public cutlass::gemm::warp::MmaTensorOp<
          Shape_, ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_,
          LayoutC_, Policy_, PartitionsK_, AccumulatorsInRowMajor> {
public:
  using Base = cutlass::gemm::warp::MmaTensorOp<
      Shape_, ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_,
      Policy_, PartitionsK_, AccumulatorsInRowMajor>;

  using Shape = Shape_;
  using Policy = Policy_;
  using ArchMmaOperator = typename Policy::Operator;

  using IteratorA = KDeepMultiplicandTileIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA_,
      LayoutA_,
      cutlass::MatrixShape<ArchMmaOperator::Shape::kM,
                           ArchMmaOperator::Shape::kK>,
      Policy::OpDelta::kRow, 32, PartitionsK_>;

  using IteratorB = KDeepMultiplicandTileIterator<
      cutlass::MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB_,
      LayoutB_,
      cutlass::MatrixShape<ArchMmaOperator::Shape::kK,
                           ArchMmaOperator::Shape::kN>,
      Policy::OpDelta::kRow, 32, PartitionsK_>;

  using FragmentA = typename IteratorA::Fragment;
  using FragmentB = typename IteratorB::Fragment;
  using TransformedFragmentA =
      cutlass::Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;
  using TransformedFragmentB =
      cutlass::Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

  static_assert(cutlass::platform::is_same<FragmentA,
                                           typename Base::FragmentA>::value &&
                    cutlass::platform::is_same<
                        FragmentB, typename Base::FragmentB>::value,
                "The reordered fragments must keep the stock storage type.");

  CUTLASS_DEVICE
  MmaTensorOpKDeep() {}
};

} // namespace dg_f64_kdeep

namespace cutlass {
namespace gemm {
namespace warp {

//- Route the K-deep f64 instruction shapes to the iterators above. The Policy
//- is the one the primary template would have built.
#define DG_F64_KDEEP_DEFAULT_MMA_TENSOR_OP(INSTRUCTION_K)                      \
  template <typename WarpShape_, typename LayoutA, typename LayoutB,           \
            typename LayoutC, int PartitionsK, bool AccumulatorsInRowMajor>    \
  struct DefaultMmaTensorOp<                                                   \
      WarpShape_, GemmShape<16, 8, INSTRUCTION_K>, double, LayoutA, double,    \
      LayoutB, double, LayoutC, arch::OpMultiplyAdd, PartitionsK,              \
      AccumulatorsInRowMajor> {                                                \
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<                     \
        cutlass::arch::Mma<GemmShape<16, 8, INSTRUCTION_K>, 32, double,        \
                           cutlass::layout::RowMajor, double,                  \
                           cutlass::layout::ColumnMajor, double,               \
                           cutlass::layout::RowMajor, arch::OpMultiplyAdd>,    \
        cutlass::MatrixShape<1, 1>>;                                           \
                                                                               \
    using Type = dg_f64_kdeep::MmaTensorOpKDeep<                               \
        WarpShape_, double, LayoutA, double, LayoutB, double, LayoutC, Policy, \
        PartitionsK, AccumulatorsInRowMajor>;                                  \
  };

DG_F64_KDEEP_DEFAULT_MMA_TENSOR_OP(8)
DG_F64_KDEEP_DEFAULT_MMA_TENSOR_OP(16)

#undef DG_F64_KDEEP_DEFAULT_MMA_TENSOR_OP

} // namespace warp
} // namespace gemm
} // namespace cutlass
