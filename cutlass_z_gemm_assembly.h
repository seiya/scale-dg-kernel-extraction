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
// Escale_y in their own epilogues (PointwiseScaleV, cuda_cutlass_gemm_fused.cu)
// and whether the y GEMM also added deriv_x into deriv_y
// (cutlass_y_gemm_scaleadd.h). When they have, this epilogue reads two volume
// tensors instead of five. Nq <= 64 uses the same kWeighted path after the y
// epilogue does Ey*acc + Ex*Dx; its x GEMM is still cuBLAS. GemmZWide stays
// off at Nq = 64 (that change alone is +2.8%).
//
// This epilogue is instruction-issue bound: removing the lift drops its
// instruction count by 13.0% and its duration by 12.8%, while the stall
// breakdown and the occupancy do not move (ncu job 63994). Everything
// kWeighted selects is therefore an instruction-count change:
//   - three volume tensors instead of five (19.9 us),
//   - the index clamps hoisted onto the tile origin (8.9 us),
//   - the six per-element lift loads issued as three 16-byte loads (9.8 us),
//   - deriv_x folded into deriv_y by the y GEMM, so two volume tensors
//     instead of three (15.7 us of this kernel at p=127),
//   - the lift's arithmetic moved behind the accumulator's shared round trip
//     so its two face loads are in flight across the barriers (5 us),
//   - 16-byte epilogue accesses (7.2 us at p=127, 17 us at p=255; at Nq = 64
//     the same change costs 2.8%, which is why GemmZWide exists),
//     which is why elembnd_flux_kernel interleaves the face planes in pairs
//     and why the two z-face lift coefficients arrive in their own packed
//     table.
// kWeighted and GemmZWide used to ride together. They no longer do: at Nq = 64
// kWeighted is on (y folds Ex*Dx) and GemmZWide stays off.
//
// The user problem is (m=Nq*Nq, n=Nq) column-major, which CUTLASS solves as
// the transposed row-major problem. So an epilogue tile row is the z index k
// and an epilogue tile column is the xy-plane index p = i + j*Nq. That is what
// lets the lift be reconstructed here without a volume-sized intermediate:
// the six face planes total 6*Nq*Nq doubles, against Nq^3 for lift_out.

//- kAffine and kPaired default to kWeighted, which is how every order
//- above Nq = 64 uses them.  They are separate template parameters so the
//- three ingredients of the package -- reading two volume tensors instead
//- of five (kWeighted), hoisting the index clamps (kAffine) and issuing the
//- six face loads as three 16-byte loads (kPaired) -- can be measured one
//- at a time at a low order.  kPaired requires the interleaved face layout
//- (pair_nq2 in elembnd_flux_kernel) and the packed lift_zpair table.
template <typename Epilogue, bool kWeighted, bool kAffine = kWeighted,
          bool kPaired = kWeighted>
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
    //- Elements per contiguous access of the output tile. The tile's
    //- contiguous direction is the xy-plane index p, so a V-wide access covers
    //- p .. p+V-1 and the lift needs its own i, j and z-face pair for each.
    static int const kEPV = ThreadMap::kElementsPerAccess;
    static int const kCols = ThreadMap::Iterations::kColumn * kEPV;

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    SharedLoadIterator sli(this->shared_storage_.reference(), thread_idx_);

    //- Clamping every row and column index keeps the addresses inside the
    //- problem when a tile hangs off the end. Clamping the tile origin once
    //- instead does the same -- an out-of-range row still reads a valid
    //- address, just not the one it would have, and the value is never stored
    //- because the output iterator predicates the store -- and it leaves the
    //- four face gathers affine in the row offset, so they strength-reduce.
    //- Worth 8.9 us per stage above Nq = 64. It is not free code: on its own
    //- it makes ptxas produce a 2.7% slower kernel at Nq = 64 and a 2.8%
    //- slower one at Nq = 8, for a bit-identical result. At Nq = 8 it is
    //- nonetheless adopted, because on top of the paired face loads and the
    //- 16-byte epilogue it turns -2.9% into -3.9%.
    static bool const kAffineIndex = kAffine;

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
    int col_i[kCols];
    int col_j[kCols];
    double col_zx[kCols];
    double col_zy[kCols];
    double col_lx1[kCols];
    double col_lx2[kCols];
    double col_ly1[kCols];
    double col_ly2[kCols];
    const int kMaxRowOffset = (ThreadMap::Iterations::kRow - 1) * ThreadMap::Delta::kRow +
                              (ThreadMap::Iterations::kGroup - 1) * ThreadMap::Delta::kGroup +
                              (ThreadMap::Iterations::kCluster - 1) * ThreadMap::Delta::kCluster;
    const int kMaxColOffset =
        (ThreadMap::Iterations::kColumn - 1) * ThreadMap::Delta::kColumn + (kEPV - 1);
    const int row_limit = max(Nq - 1 - kMaxRowOffset, 0);
    const int col_limit = max(nq2 - 1 - kMaxColOffset, 0);

    {
      const int start_col = kAffineIndex
                                ? min(destination_iterator.thread_start_column(), col_limit)
                                : destination_iterator.thread_start_column();
      CUTLASS_PRAGMA_UNROLL
      for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kEPV; ++e) {
          const int cc = column * kEPV + e;
          const int c = start_col + column * ThreadMap::Delta::kColumn + e;
          const int p = kAffineIndex ? c : min(c, nq2 - 1);
          const int i = p % Nq;
          const int j = p / Nq;
          col_i[cc] = i;
          col_j[cc] = j;
          if (kPaired) {
            const double2 z = pC[p];
            col_zx[cc] = z.x;
            col_zy[cc] = z.y;
          } else {
            col_zx[cc] = fb4[p];
            col_zy[cc] = fb5[p];
          }
          col_lx1[cc] = lift1d[Nq + i];
          col_lx2[cc] = lift1d[3 * Nq + i];
          col_ly1[cc] = lift1d[j];
          col_ly2[cc] = lift1d[2 * Nq + j];
        }
      }
    }

    #pragma unroll(1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      typename OutputTileIterator::Fragment frag_dx, frag_dy, frag_ex, frag_ey, frag_ez;
      if (kWeighted) {
        //- One volume tensor, not two: the y GEMM added deriv_x into deriv_y.
        it_dx.load(frag_dx);
        it_ez.load(frag_ez);
        ++it_dx;
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
      double2 lift_a[OutputTileIterator::Fragment::kElements];
      double2 lift_b[OutputTileIterator::Fragment::kElements];
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
            const double2 lzp = kPaired ? pz[k] : make_double2(0.0, 0.0);
            const double lz1 = kPaired ? lzp.x : lift1d[4 * Nq + k];
            const double lz2 = kPaired ? lzp.y : lift1d[5 * Nq + k];
            const int kNq = k * Nq;

            CUTLASS_PRAGMA_UNROLL
            for (int cc = 0; cc < kCols; ++cc) {
              const int idx = frag_row_idx * kCols + cc;
              if (kPaired) {
                //- Load phase only: keep the two face pairs in registers and
                //- let the accumulator's shared round trip below cover their
                //- latency.  The arithmetic runs after sli.load().
                lift_b[idx] = pB[col_j[cc] + kNq];
                lift_a[idx] = pA[col_i[cc] + kNq];
                frag_lift[idx] = lz1 * col_zx[cc] + lz2 * col_zy[cc];
                continue;
              }
              const double lx = col_lx1[cc] * fb1[col_j[cc] + kNq] +
                                col_lx2[cc] * fb3[col_j[cc] + kNq];
              const double ly = col_ly1[cc] * fb0[col_i[cc] + kNq] +
                                col_ly2[cc] * fb2[col_i[cc] + kNq];
              const double lz = lz1 * col_zx[cc] + lz2 * col_zy[cc];
              frag_lift[idx] = (lx + ly) + lz;
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

      //- Arithmetic phase of the lift, deliberately after the accumulator's
      //- shared round trip so that the two face loads above are in flight
      //- across the two barriers instead of being waited on immediately.
      //- Worth 5 us per stage at p=127 for a bit-identical result.
      if (kPaired) {
        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
          CUTLASS_PRAGMA_UNROLL
          for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
            CUTLASS_PRAGMA_UNROLL
            for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
              const int frag_row_idx =
                  row + ThreadMap::Iterations::kRow *
                            (group + ThreadMap::Iterations::kGroup * cluster);
              CUTLASS_PRAGMA_UNROLL
              for (int cc = 0; cc < kCols; ++cc) {
                const int idx = frag_row_idx * kCols + cc;
                const double lx = col_lx1[cc] * lift_b[idx].x +
                                  col_lx2[cc] * lift_b[idx].y;
                const double ly = col_ly1[cc] * lift_a[idx].x +
                                  col_ly2[cc] * lift_a[idx].y;
                frag_lift[idx] = (lx + ly) + frag_lift[idx];
              }
            }
          }
        }
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
          //- dx already holds deriv_x + deriv_y; see cutlass_y_gemm_scaleadd.h.
          out[i] = -(dx[i] + ez[i] * dz[i] + lf[i]);
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

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_, bool kWeighted,
          bool kAffine = kWeighted, bool kPaired = kWeighted>
struct GemmBatchedDqdtAssembly {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using BaseKernel = cutlass::gemm::kernel::GemmBatched<Mma, Epilogue, ThreadblockSwizzle>;
  using AssemblyEpilogue = EpilogueDqdtAssembly<Epilogue, kWeighted, kAffine, kPaired>;
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
