#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/arch/synclog.hpp"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/kernel/gemm_batched.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"

// z-volume GEMM with the standard TensorOp mainloop. After the accumulator
// tile is staged through shared memory, the epilogue loads Dx, Dy and the
// Escale fields it still needs, evaluates the separable surface lift directly
// from the six face planes, and stores
//   dqdt = -(Ex*Dx + Ey*Dy + Ez*Dz + lift)
// instead of writing Dz.
//
// kWeighted says whether the x and y GEMMs already applied Escale_x and
// Escale_y in their own epilogues (PointwiseScale, cuda_cutlass_gemm_fused.cu).
// When they have, this epilogue reads three volume tensors instead of five,
// which is worth 19.9 us per stage at p=255. It is false on the Nq <= 64
// branch, whose x GEMM is the cuBLAS one and has no epilogue to weight in.
//
// This epilogue is instruction-issue bound: removing the lift drops its
// instruction count by 13.0% and its duration by 12.8%, while the stall
// breakdown and the occupancy do not move (ncu job 63994). Everything
// kWeighted selects is therefore an instruction-count change:
//   - three volume tensors instead of five (19.9 us),
//   - the index clamps hoisted onto the tile origin (8.9 us),
//   - the six per-element lift loads issued as three 16-byte loads (9.8 us),
//     which is why elembnd_flux_kernel interleaves the face planes in pairs
//     and why the two z-face lift coefficients arrive in their own packed
//     table.
// They ride on one template parameter because they stand or fall together:
// the shallow branch runs its x GEMM on cuBLAS, which has no epilogue to
// weight Escale into, and measuring the other two there gave +0.8% and +2.7%
// for bit-identical results.
//
// The user problem is (m=Nq*Nq, n=Nq) column-major, which CUTLASS solves as
// the transposed row-major problem. So an epilogue tile row is the z index k
// and an epilogue tile column is the xy-plane index p = i + j*Nq. That is what
// lets the lift be reconstructed here without a volume-sized intermediate:
// the six face planes total 6*Nq*Nq doubles, against Nq^3 for lift_out.

template <typename Epilogue, bool kWeighted>
class EpilogueDqdtAssembly : public Epilogue {
public:
  using OutputTileIterator = typename Epilogue::OutputTileIterator;
  using AccumulatorTile = typename Epilogue::AccumulatorTile;
  using AccumulatorFragmentIterator = typename Epilogue::AccumulatorFragmentIterator;
  using SharedLoadIterator = typename Epilogue::SharedLoadIterator;
  using WarpTileIterator = typename Epilogue::WarpTileIterator;
  using SharedStorage = typename Epilogue::SharedStorage;
  static int const kPartitionsK = Epilogue::kPartitionsK;
  static int constexpr kSmemPointerOffset = Epilogue::kSmemPointerOffset;

  CUTLASS_DEVICE
  EpilogueDqdtAssembly(SharedStorage &shared_storage, int thread_idx, int warp_idx,
                       int lane_idx)
      : Epilogue(shared_storage, thread_idx, warp_idx, lane_idx), thread_idx_(thread_idx)
  {
  }

private:
  int thread_idx_;

public:

  template <class Seq>
  struct acc2smem;

  template <size_t... Seq>
  struct acc2smem<cutlass::index_sequence<Seq...>> {
    template <int Advance>
    CUTLASS_DEVICE
    static void helper(AccumulatorFragmentIterator accum_fragment_iterator,
                       WarpTileIterator &warp_tile_iterator)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }
      typename AccumulatorFragmentIterator::Fragment accum_fragment;
      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;
      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void push(size_t pos, AccumulatorFragmentIterator const &iterator_begin,
                     WarpTileIterator &warp_tile_iterator)
    {
      int dummy[] = {(pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
      (void)dummy;
    }
  };

  CUTLASS_DEVICE
  void apply_assembly(OutputTileIterator destination_iterator, AccumulatorTile const &accumulators,
                      OutputTileIterator it_dx, OutputTileIterator it_dy,
                      OutputTileIterator it_ex, OutputTileIterator it_ey,
                      OutputTileIterator it_ez, double const *flux_bnd,
                      double const *lift1d, double const *lift_zpair, int Nq)
  {
    using ThreadMap = typename OutputTileIterator::ThreadMap;
    static_assert(ThreadMap::kElementsPerAccess == 1,
                  "the lift reconstruction assumes one element per access");

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    SharedLoadIterator sli(this->shared_storage_.reference(), thread_idx_);

    //- Clamping every row and column index keeps the addresses inside the
    //- problem when a tile hangs off the end. Clamping the tile origin once
    //- instead does the same -- an out-of-range row still reads a valid
    //- address, just not the one it would have, and the value is never stored
    //- because the output iterator predicates the store -- and it leaves the
    //- four face gathers affine in the row offset, so they strength-reduce.
    //- Worth 8.9 us per stage. Only on the kWeighted branch: the same change
    //- makes ptxas produce a 2.7% slower kernel on the other one, for a
    //- bit-identical result.
    static bool const kAffineIndex = kWeighted;

    const int nq2 = Nq * Nq;
    double const *fb0 = flux_bnd;
    double const *fb1 = flux_bnd + nq2;
    double const *fb2 = flux_bnd + 2 * nq2;
    double const *fb3 = flux_bnd + 3 * nq2;
    double const *fb4 = flux_bnd + 4 * nq2;
    double const *fb5 = flux_bnd + 5 * nq2;
    //- Paired view of the same buffer when kWeighted: faces 1 and 3 in
    //- [0, 2*nq2), faces 2 and 4 next, faces 5 and 6 last. The two members of
    //- every pair are the two the lift reads at the same index.
    double2 const *pA = reinterpret_cast<double2 const *>(flux_bnd);
    double2 const *pB = reinterpret_cast<double2 const *>(flux_bnd + 2 * nq2);
    double2 const *pC = reinterpret_cast<double2 const *>(flux_bnd + 4 * nq2);
    double2 const *pz = reinterpret_cast<double2 const *>(lift_zpair);

    // operator++ advances only thread_start_row_, so everything that depends
    // on the tile column -- and with it the integer division by Nq and the two
    // z-face values -- is invariant across the kIterations loop below.
    int col_i[ThreadMap::Iterations::kColumn];
    int col_j[ThreadMap::Iterations::kColumn];
    double col_zx[ThreadMap::Iterations::kColumn];
    double col_zy[ThreadMap::Iterations::kColumn];
    double col_lx1[ThreadMap::Iterations::kColumn];
    double col_lx2[ThreadMap::Iterations::kColumn];
    double col_ly1[ThreadMap::Iterations::kColumn];
    double col_ly2[ThreadMap::Iterations::kColumn];
    const int kMaxRowOffset = (ThreadMap::Iterations::kRow - 1) * ThreadMap::Delta::kRow +
                              (ThreadMap::Iterations::kGroup - 1) * ThreadMap::Delta::kGroup +
                              (ThreadMap::Iterations::kCluster - 1) * ThreadMap::Delta::kCluster;
    const int kMaxColOffset = (ThreadMap::Iterations::kColumn - 1) * ThreadMap::Delta::kColumn;
    const int row_limit = max(Nq - 1 - kMaxRowOffset, 0);
    const int col_limit = max(nq2 - 1 - kMaxColOffset, 0);

    {
      const int start_col = kAffineIndex
                                ? min(destination_iterator.thread_start_column(), col_limit)
                                : destination_iterator.thread_start_column();
      CUTLASS_PRAGMA_UNROLL
      for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
        const int c = start_col + column * ThreadMap::Delta::kColumn;
        const int p = kAffineIndex ? c : min(c, nq2 - 1);
        const int i = p % Nq;
        const int j = p / Nq;
        col_i[column] = i;
        col_j[column] = j;
        if (kWeighted) {
          const double2 z = pC[p];
          col_zx[column] = z.x;
          col_zy[column] = z.y;
        } else {
          col_zx[column] = fb4[p];
          col_zy[column] = fb5[p];
        }
        col_lx1[column] = lift1d[Nq + i];
        col_lx2[column] = lift1d[3 * Nq + i];
        col_ly1[column] = lift1d[j];
        col_ly2[column] = lift1d[2 * Nq + j];
      }
    }

    #pragma unroll(1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      typename OutputTileIterator::Fragment frag_dx, frag_dy, frag_ex, frag_ey, frag_ez;
      if (kWeighted) {
        it_dx.load(frag_dx);
        it_dy.load(frag_dy);
        it_ez.load(frag_ez);
        ++it_dx;
        ++it_dy;
        ++it_ez;
      } else {
        it_dx.load(frag_dx);
        it_dy.load(frag_dy);
        it_ex.load(frag_ex);
        it_ey.load(frag_ey);
        it_ez.load(frag_ez);
        ++it_dx;
        ++it_dy;
        ++it_ex;
        ++it_ey;
        ++it_ez;
      }

      // lift(i,j,k), summed in the same (x+y)+z order the three K=2 GEMMs used.
      double frag_lift[OutputTileIterator::Fragment::kElements];
      const int start_row = kAffineIndex
                                ? min(destination_iterator.thread_start_row(), row_limit)
                                : destination_iterator.thread_start_row();
      CUTLASS_PRAGMA_UNROLL
      for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
        CUTLASS_PRAGMA_UNROLL
        for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
          CUTLASS_PRAGMA_UNROLL
          for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            const int frag_row_idx =
                row + ThreadMap::Iterations::kRow *
                          (group + ThreadMap::Iterations::kGroup * cluster);
            const int row_offset = row * ThreadMap::Delta::kRow +
                                   group * ThreadMap::Delta::kGroup +
                                   cluster * ThreadMap::Delta::kCluster;
            const int k = kAffineIndex ? (start_row + row_offset)
                                       : min(start_row + row_offset, Nq - 1);
            const double2 lzp = kWeighted ? pz[k] : make_double2(0.0, 0.0);
            const double lz1 = kWeighted ? lzp.x : lift1d[4 * Nq + k];
            const double lz2 = kWeighted ? lzp.y : lift1d[5 * Nq + k];
            const int kNq = k * Nq;

            CUTLASS_PRAGMA_UNROLL
            for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
              double lx, ly;
              if (kWeighted) {
                const double2 b = pB[col_j[column] + kNq];
                const double2 a = pA[col_i[column] + kNq];
                lx = col_lx1[column] * b.x + col_lx2[column] * b.y;
                ly = col_ly1[column] * a.x + col_ly2[column] * a.y;
              } else {
                lx = col_lx1[column] * fb1[col_j[column] + kNq] +
                     col_lx2[column] * fb3[col_j[column] + kNq];
                ly = col_ly1[column] * fb0[col_i[column] + kNq] +
                     col_ly2[column] * fb2[col_i[column] + kNq];
              }
              const double lz = lz1 * col_zx[column] + lz2 * col_zy[column];
              frag_lift[frag_row_idx * ThreadMap::Iterations::kColumn + column] =
                  (lx + ly) + lz;
            }
          }
        }
      }

      __syncthreads();
      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
          iter, accum_fragment_iterator, this->warp_tile_iterator_);
      __syncthreads();

      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
      sli.load(aligned_accum_fragment[0]);
      if (kPartitionsK > 1) {
        cutlass::plus<typename SharedLoadIterator::Fragment> add_fragments;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kPartitionsK; ++i) {
          sli.add_pointer_offset(kSmemPointerOffset);
          sli.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0],
                                                    aligned_accum_fragment[i]);
        }
        sli.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
      }

      typename OutputTileIterator::Fragment output_fragment;
      auto const *dz = reinterpret_cast<double const *>(&aligned_accum_fragment[0]);
      auto const *dx = reinterpret_cast<double const *>(&frag_dx);
      auto const *dy = reinterpret_cast<double const *>(&frag_dy);
      auto const *lf = frag_lift;
      auto const *ex = reinterpret_cast<double const *>(&frag_ex);
      auto const *ey = reinterpret_cast<double const *>(&frag_ey);
      auto const *ez = reinterpret_cast<double const *>(&frag_ez);
      auto *out = reinterpret_cast<double *>(&output_fragment);
      if (kWeighted) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < OutputTileIterator::Fragment::kElements; ++i) {
          out[i] = -(dx[i] + dy[i] + ez[i] * dz[i] + lf[i]);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < OutputTileIterator::Fragment::kElements; ++i) {
          out[i] = -(ex[i] * dx[i] + ey[i] * dy[i] + ez[i] * dz[i] + lf[i]);
        }
      }

      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
  }
};

// The accumulator tile is staged through shared memory before the epilogue
// reads it back in the output layout. DefaultEpilogueTensorOp pads that tile by
// 64 / sizeof_bits<double> * 4 = 4 doubles, so a row is 36 doubles = 72 words
// wide. One STS.128 phase covers eight lanes, i.e. two consecutive rows, and
// 72 mod 32 = 8, so the second row starts eight banks into the first row's
// sixteen: every accumulator store is a 2-way conflict and costs eight
// wavefronts instead of four.
//
// A row stride of 8 doubles mod 16 puts consecutive rows exactly sixteen banks
// apart, so the two rows of a phase tile the 32 banks once. Padding by 8 gives
// that for the 32-, 64- and 128-column threadblock tiles alike. The epilogue
// tile shares a union with the mainloop's 48 KB, so this costs no shared
// memory and no occupancy.
template <typename Epilogue_, int PaddingColumn>
using RepadEpilogue = cutlass::epilogue::threadblock::Epilogue<
    typename Epilogue_::Shape,
    typename Epilogue_::WarpMmaOperator,
    Epilogue_::kPartitionsK,
    typename Epilogue_::OutputTileIterator,
    typename Epilogue_::AccumulatorFragmentIterator,
    typename Epilogue_::WarpTileIterator,
    typename Epilogue_::SharedLoadIterator,
    typename Epilogue_::OutputOp,
    cutlass::MatrixShape<0, PaddingColumn>,
    Epilogue_::kFragmentsPerIteration>;

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_, bool kWeighted>
struct GemmBatchedDqdtAssembly {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using BaseKernel = cutlass::gemm::kernel::GemmBatched<Mma, Epilogue, ThreadblockSwizzle>;
  using AssemblyEpilogue = EpilogueDqdtAssembly<Epilogue, kWeighted>;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Params {
    typename BaseKernel::Params gemm{};
    double const *ptr_dx{nullptr};
    double const *ptr_dy{nullptr};
    double const *ptr_flux_bnd{nullptr};
    double const *ptr_lift1d{nullptr};
    double const *ptr_lift_zpair{nullptr};
    int Nq{0};
    double const *ptr_ex{nullptr};
    double const *ptr_ey{nullptr};
    double const *ptr_ez{nullptr};
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage)
  {
    typename BaseKernel::Params const &gp = params.gemm;
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(gp.swizzle_log_tile);

    if (gp.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        gp.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    for (int batch_idx = threadblock_swizzle.get_batch_idx(); batch_idx < gp.batch_count;
         batch_idx += gridDim.z) {
      cutlass::MatrixCoord tb_offset_A{threadblock_tile_offset.m() * Mma::Shape::kM, 0};
      cutlass::MatrixCoord tb_offset_B{0, threadblock_tile_offset.n() * Mma::Shape::kN};
      int thread_idx = threadIdx.x;

      typename Mma::IteratorA iterator_A(gp.params_A, gp.ref_A.data(), gp.problem_size.mk(),
                                         thread_idx, tb_offset_A);
      iterator_A.add_pointer_offset(gp.stride_A * batch_idx);

      typename Mma::IteratorB iterator_B(gp.params_B, gp.ref_B.data(), gp.problem_size.kn(),
                                         thread_idx, tb_offset_B);
      iterator_B.add_pointer_offset(gp.stride_B * batch_idx);

      int warp_idx = threadIdx.x / 32;
      int lane_idx = threadIdx.x % 32;
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
      typename Mma::FragmentC accumulators;
      accumulators.clear();
      mma(gp.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

      threadblock_tile_offset = threadblock_swizzle.get_tile_offset(gp.swizzle_log_tile);
      cutlass::MatrixCoord threadblock_offset(threadblock_tile_offset.m() * Mma::Shape::kM,
                                              threadblock_tile_offset.n() * Mma::Shape::kN);

      typename Epilogue::OutputTileIterator iterator_D(
          gp.params_D, gp.ref_D.data(), gp.problem_size.mn(), thread_idx, threadblock_offset);
      iterator_D.add_pointer_offset(gp.stride_D * batch_idx);

      typename Epilogue::OutputTileIterator it_dx(gp.params_D, const_cast<double *>(params.ptr_dx),
                                                  gp.problem_size.mn(), thread_idx,
                                                  threadblock_offset);
      it_dx.add_pointer_offset(gp.stride_D * batch_idx);
      typename Epilogue::OutputTileIterator it_dy(gp.params_D, const_cast<double *>(params.ptr_dy),
                                                  gp.problem_size.mn(), thread_idx,
                                                  threadblock_offset);
      it_dy.add_pointer_offset(gp.stride_D * batch_idx);
      typename Epilogue::OutputTileIterator it_ex(gp.params_D, const_cast<double *>(params.ptr_ex),
                                                  gp.problem_size.mn(), thread_idx,
                                                  threadblock_offset);
      it_ex.add_pointer_offset(gp.stride_D * batch_idx);
      typename Epilogue::OutputTileIterator it_ey(gp.params_D, const_cast<double *>(params.ptr_ey),
                                                  gp.problem_size.mn(), thread_idx,
                                                  threadblock_offset);
      it_ey.add_pointer_offset(gp.stride_D * batch_idx);
      typename Epilogue::OutputTileIterator it_ez(gp.params_D, const_cast<double *>(params.ptr_ez),
                                                  gp.problem_size.mn(), thread_idx,
                                                  threadblock_offset);
      it_ez.add_pointer_offset(gp.stride_D * batch_idx);

      AssemblyEpilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
      const int nfp_tot = 6 * params.Nq * params.Nq;
      epilogue.apply_assembly(iterator_D, accumulators, it_dx, it_dy, it_ex, it_ey, it_ez,
                              params.ptr_flux_bnd + int64_t(nfp_tot) * batch_idx,
                              params.ptr_lift1d, params.ptr_lift_zpair, params.Nq);
    }
  }
};
