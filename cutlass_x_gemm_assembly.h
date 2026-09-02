#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/arch/synclog.hpp"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"

#include "cutlass_z_gemm_assembly.h"   // RepadEpilogue

//- x volume GEMM carrying the final assembly epilogue, i.e. the same fusion
//- package GEMM_FUSED normally puts on z, moved onto x.  AGENTS.md leaves the
//- choice of carrier open: it only requires that the LAST volume GEMM fuse the
//- weighting and the surface lift, and that the library assignment and the
//- mainloop tiles stay the same as GEMM_CUTE's.  This header changes neither:
//- the mainloop is Tile::GemmX, exactly what GEMM_CUTE runs for x.
//
//- Geometry.  The x GEMM is a single (m = Nq, n = Nq*Nq*Ne, k = Nq) problem
//- with a column-major C, which device::Gemm solves transposed.  So an
//- epilogue tile ROW is the problem's n index c = j + Nq*k + Nq*Nq*elem (the
//- long axis) and an epilogue tile COLUMN is the problem's m index i, whose
//- extent is exactly Nq.  That is the one structural difference from the other
//- two carriers: the threadblock's N extent equals Nq at every order, so the
//- carrier tile is never half-predicated the way the y tile is at Nq = 16 and
//- the z tile is at Nq = 32.
//
//- Face loads, against the z and y carriers:
//-   z carrier (rows = k, cols = p = i + j*Nq): lz separable, lx and ly each
//-             need a per-element face load -> 4 loads/element.
//-   y carrier (rows = j, cols = i, k fixed per batch): lx and ly separable,
//-             only lz needs a per-element load -> 2 loads/element.
//-   x carrier (rows = c = (j,k,elem), cols = i): lx is separable -- its
//-             coefficient depends on i alone and its face value on (j,k)
//-             alone -- but ly needs fb0/fb2[i + k*Nq] and lz needs
//-             fb4/fb5[i + j*Nq], both of which vary in the row AND the
//-             column -> 4 loads/element, like z.
//- So x buys tile fit, not face-load count.
//
//- Volume tensors.  The accumulator is D(flux_x), so this epilogue reads Dy,
//- Dz, Ex, Ey and Ez: five, like the unweighted z carrier.  Unlike the y
//- carrier -- where Escale_y multiplies the kernel's own accumulator and can
//- never be forwarded -- BOTH of the other two Escale factors can be forwarded
//- here, because y and z are now plain GEMMs.  kYWeighted says Escale_y is
//- already in deriv_y, kZWeighted says Escale_z is already in deriv_z; with
//- both, this epilogue reads three volume tensors (Dy, Dz, Ex).
//
//- kAffine and kPaired are the two instruction-count ingredients the Nq > 64
//- z carrier uses (cutlass_z_gemm_assembly.h): hoisting the index clamps onto
//- the tile origin so the face gathers stay affine in the row offset, and
//- issuing the six per-element face loads as three 16-byte loads out of the
//- interleaved face layout (pair_nq2 in elembnd_flux_kernel) plus a packed
//- lift-coefficient table.  Both are expressible on x: the x carrier reads
//- exactly the same three face pairs -- (f1,f3) at jk, (f0,f2) at i+k*Nq and
//- (f4,f5) at i+j*Nq -- that the interleaved layout groups.  The third
//- ingredient, 16-byte epilogue accesses, is not in this header: it is the
//- output tile iterator's kElementsPerAccess and comes from an x tile built
//- with EpilogueOp2 (XTileWide in cuda_cutlass_gemm_fused.cu), same
//- threadblock/warp/stage shape as the plain tile so GEMM_CUTE still shares
//- the mainloop.
template <typename Epilogue, bool kYWeighted = false, bool kZWeighted = false,
          bool kAffine = false, bool kPaired = false>
class EpilogueDqdtAssemblyX : public Epilogue {
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
  EpilogueDqdtAssemblyX(SharedStorage &shared_storage, int thread_idx, int warp_idx,
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
                      OutputTileIterator it_dy, OutputTileIterator it_dz,
                      OutputTileIterator it_ex, OutputTileIterator it_ey,
                      OutputTileIterator it_ez, double const *flux_bnd,
                      double const *lift1d, double const *lift_pair, int Nq,
                      int nrow, int nq_log2)
  {
    using ThreadMap = typename OutputTileIterator::ThreadMap;
    static int const kEPV = ThreadMap::kElementsPerAccess;
    static int const kCols = ThreadMap::Iterations::kColumn * kEPV;

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    SharedLoadIterator sli(this->shared_storage_.reference(), thread_idx_);

    const int nq2 = Nq * Nq;
    const int nfp_tot = 6 * nq2;

    //- Packed lift coefficient pairs, layout as pack_lift_pair_kernel writes
    //- them: [0,2Nq) = (Lift1D(:,1), Lift1D(:,3)) indexed by j, [2Nq,4Nq) =
    //- (Lift1D(:,2), Lift1D(:,4)) indexed by i, [4Nq,6Nq) = (Lift1D(:,5),
    //- Lift1D(:,6)) indexed by k.
    double2 const *py = reinterpret_cast<double2 const *>(lift_pair);
    double2 const *px = reinterpret_cast<double2 const *>(lift_pair + 2 * Nq);
    double2 const *pz = reinterpret_cast<double2 const *>(lift_pair + 4 * Nq);

    //- Clamping the tile ORIGIN instead of every index: an out-of-range row
    //- still reads a valid address and the output iterator predicates the
    //- store away, so the result is unchanged while the face gathers stay
    //- affine in the row offset.  Same trade as the z carrier's kAffine.
    const int kMaxRowOffset = (ThreadMap::Iterations::kRow - 1) * ThreadMap::Delta::kRow +
                              (ThreadMap::Iterations::kGroup - 1) * ThreadMap::Delta::kGroup +
                              (ThreadMap::Iterations::kCluster - 1) * ThreadMap::Delta::kCluster;
    const int kMaxColOffset =
        (ThreadMap::Iterations::kColumn - 1) * ThreadMap::Delta::kColumn + (kEPV - 1);
    const int row_limit = max(nrow - 1 - kMaxRowOffset, 0);
    const int col_limit = max(Nq - 1 - kMaxColOffset, 0);

    //- Column-invariant part: i and the two x-face lift coefficients, which
    //- depend on i alone.  operator++ on the output iterator advances only
    //- thread_start_row_, so this survives the kIterations loop.
    int col_i[kCols];
    double col_lx1[kCols];
    double col_lx2[kCols];
    {
      const int start_col = kAffine ? min(destination_iterator.thread_start_column(), col_limit)
                                    : destination_iterator.thread_start_column();
      CUTLASS_PRAGMA_UNROLL
      for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kEPV; ++e) {
          const int cc = column * kEPV + e;
          const int c = start_col + column * ThreadMap::Delta::kColumn + e;
          const int i = kAffine ? c : min(c, Nq - 1);
          col_i[cc] = i;
          if (kPaired) {
            const double2 lxp = px[i];
            col_lx1[cc] = lxp.x;
            col_lx2[cc] = lxp.y;
          } else {
            col_lx1[cc] = lift1d[Nq + i];
            col_lx2[cc] = lift1d[3 * Nq + i];
          }
        }
      }
    }

    #pragma unroll(1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      typename OutputTileIterator::Fragment frag_dy, frag_dz, frag_ex, frag_ey, frag_ez;
      it_dy.load(frag_dy);
      it_dz.load(frag_dz);
      it_ex.load(frag_ex);
      ++it_dy;
      ++it_dz;
      ++it_ex;
      if (!kYWeighted) {
        it_ey.load(frag_ey);
        ++it_ey;
      }
      if (!kZWeighted) {
        it_ez.load(frag_ez);
        ++it_ez;
      }

      // lift(i,j,k), summed in the same (x+y)+z order the two-kernel form used.
      double frag_lift[OutputTileIterator::Fragment::kElements];
      double2 lift_a[kPaired ? OutputTileIterator::Fragment::kElements : 1];
      double2 lift_c[kPaired ? OutputTileIterator::Fragment::kElements : 1];
      static int const kFragRows = ThreadMap::Iterations::kRow *
                                   ThreadMap::Iterations::kGroup *
                                   ThreadMap::Iterations::kCluster;
      double row_ly1[kPaired ? kFragRows : 1];
      double row_ly2[kPaired ? kFragRows : 1];
      double row_lz1[kPaired ? kFragRows : 1];
      double row_lz2[kPaired ? kFragRows : 1];
      const int start_row = kAffine ? min(destination_iterator.thread_start_row(), row_limit)
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
            //- The row is the packed index c = j + Nq*k + Nq*Nq*elem.  Every
            //- adopted x tile has TbM dividing Nq*Nq, so elem is in fact
            //- constant over a tile, but nothing here relies on that.
            const int c = kAffine ? (start_row + row_offset)
                                  : min(start_row + row_offset, nrow - 1);
            //- Splitting c costs two integer divisions per row, and unlike the
            //- z and y carriers the x carrier's rows ARE the long axis, so it
            //- pays them once per row iteration instead of hoisting them into
            //- the column precompute.  nq_log2 >= 0 (every power-of-two Nq)
            //- replaces them with shifts and masks; passing -1 is the general
            //- fallback and the A/B that prices the arithmetic.
            int elem, j, kz;
            if (nq_log2 >= 0) {
              elem = c >> (2 * nq_log2);
              const int rem = c & (nq2 - 1);
              j = rem & (Nq - 1);
              kz = rem >> nq_log2;
            } else {
              elem = c / nq2;
              const int rem = c - elem * nq2;
              j = rem % Nq;
              kz = rem / Nq;
            }

            double const *fb = flux_bnd + static_cast<std::int64_t>(nfp_tot) * elem;
            const int jk = j + kz * Nq;
            const int kNq = kz * Nq;
            const int jNq = j * Nq;

            if (kPaired) {
              //- Interleaved face layout: pA = (f0,f2), pB = (f1,f3),
              //- pC = (f4,f5), each at the index the lift reads them at.
              double2 const *pA = reinterpret_cast<double2 const *>(fb);
              double2 const *pB = reinterpret_cast<double2 const *>(fb + 2 * nq2);
              double2 const *pC = reinterpret_cast<double2 const *>(fb + 4 * nq2);
              const double2 fxp = pB[jk];
              const double2 lyp = py[j];
              const double2 lzp = pz[kz];
              row_ly1[frag_row_idx] = lyp.x;
              row_ly2[frag_row_idx] = lyp.y;
              row_lz1[frag_row_idx] = lzp.x;
              row_lz2[frag_row_idx] = lzp.y;
              CUTLASS_PRAGMA_UNROLL
              for (int cc = 0; cc < kCols; ++cc) {
                const int idx = frag_row_idx * kCols + cc;
                const int i = col_i[cc];
                //- Load phase only: keep the two per-element face pairs in
                //- registers and let the accumulator's shared round trip
                //- below cover their latency.  The arithmetic runs after
                //- sli.load(), in the same (lx+ly)+lz order.
                lift_a[idx] = pA[i + kNq];
                lift_c[idx] = pC[i + jNq];
                frag_lift[idx] = col_lx1[cc] * fxp.x + col_lx2[cc] * fxp.y;
              }
              continue;
            }

            double const *fb0 = fb;
            double const *fb1 = fb + nq2;
            double const *fb2 = fb + 2 * nq2;
            double const *fb3 = fb + 3 * nq2;
            double const *fb4 = fb + 4 * nq2;
            double const *fb5 = fb + 5 * nq2;

            //- Row-invariant scalars: the two x-face VALUES (they depend on j
            //- and k only, which is what makes lx separable here), and the y-
            //- and z-face lift coefficients.
            const double fx1 = fb1[jk];
            const double fx3 = fb3[jk];
            const double ly1 = lift1d[j];
            const double ly2 = lift1d[2 * Nq + j];
            const double lz1 = lift1d[4 * Nq + kz];
            const double lz2 = lift1d[5 * Nq + kz];

            CUTLASS_PRAGMA_UNROLL
            for (int cc = 0; cc < kCols; ++cc) {
              const int idx = frag_row_idx * kCols + cc;
              const int i = col_i[cc];
              const double lx = col_lx1[cc] * fx1 + col_lx2[cc] * fx3;
              const double ly = ly1 * fb0[i + kNq] + ly2 * fb2[i + kNq];
              const double lz = lz1 * fb4[i + jNq] + lz2 * fb5[i + jNq];
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
      //- shared round trip so the two per-element face loads above are in
      //- flight across the two barriers.  Same (lx+ly)+lz order: frag_lift
      //- already holds lx.
      if (kPaired) {
        CUTLASS_PRAGMA_UNROLL
        for (int fr = 0; fr < kFragRows; ++fr) {
          const double ly1 = row_ly1[fr];
          const double ly2 = row_ly2[fr];
          const double lz1 = row_lz1[fr];
          const double lz2 = row_lz2[fr];
          CUTLASS_PRAGMA_UNROLL
          for (int cc = 0; cc < kCols; ++cc) {
            const int idx = fr * kCols + cc;
            const double ly = ly1 * lift_a[idx].x + ly2 * lift_a[idx].y;
            const double lz = lz1 * lift_c[idx].x + lz2 * lift_c[idx].y;
            frag_lift[idx] = (frag_lift[idx] + ly) + lz;
          }
        }
      }

      typename OutputTileIterator::Fragment output_fragment;
      auto const *dx = reinterpret_cast<double const *>(&aligned_accum_fragment[0]);
      auto const *dy = reinterpret_cast<double const *>(&frag_dy);
      auto const *dz = reinterpret_cast<double const *>(&frag_dz);
      auto const *lf = frag_lift;
      auto const *ex = reinterpret_cast<double const *>(&frag_ex);
      auto const *ey = reinterpret_cast<double const *>(&frag_ey);
      auto const *ez = reinterpret_cast<double const *>(&frag_ez);
      auto *out = reinterpret_cast<double *>(&output_fragment);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < OutputTileIterator::Fragment::kElements; ++i) {
        const double ty = kYWeighted ? dy[i] : ey[i] * dy[i];
        const double tz = kZWeighted ? dz[i] : ez[i] * dz[i];
        out[i] = -(ex[i] * dx[i] + ty + tz + lf[i]);
      }

      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
  }
};

//- Non-batched counterpart of GemmBatchedDqdtAssembly / ...Y: the x volume GEMM
//- is a single large problem, not a batch, so this wraps kernel::Gemm.
template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_,
          bool kYWeighted = false, bool kZWeighted = false,
          bool kAffine = false, bool kPaired = false>
struct GemmDqdtAssemblyX {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using BaseKernel =
      cutlass::gemm::kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, false>;
  using AssemblyEpilogue =
      EpilogueDqdtAssemblyX<Epilogue, kYWeighted, kZWeighted, kAffine, kPaired>;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Params {
    typename BaseKernel::Params gemm{};
    double const *ptr_dy{nullptr};
    double const *ptr_dz{nullptr};
    double const *ptr_flux_bnd{nullptr};
    double const *ptr_lift1d{nullptr};
    double const *ptr_lift_pair{nullptr};
    int Nq{0};
    int nq_log2{-1};
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

    cutlass::MatrixCoord tb_offset_A{threadblock_tile_offset.m() * Mma::Shape::kM,
                                     threadblock_tile_offset.k() * gp.gemm_k_size};
    cutlass::MatrixCoord tb_offset_B{threadblock_tile_offset.k() * gp.gemm_k_size,
                                     threadblock_tile_offset.n() * Mma::Shape::kN};

    int problem_size_k =
        min(gp.problem_size.k(), (threadblock_tile_offset.k() + 1) * gp.gemm_k_size);
    int gemm_k_iterations =
        (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    int thread_idx = threadIdx.x;

    typename Mma::IteratorA iterator_A(gp.params_A, gp.ref_A.data(),
                                       {gp.problem_size.m(), problem_size_k}, thread_idx,
                                       tb_offset_A);
    typename Mma::IteratorB iterator_B(gp.params_B, gp.ref_B.data(),
                                       {problem_size_k, gp.problem_size.n()}, thread_idx,
                                       tb_offset_B);

    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(gp.swizzle_log_tile);
    cutlass::MatrixCoord threadblock_offset(threadblock_tile_offset.m() * Mma::Shape::kM,
                                            threadblock_tile_offset.n() * Mma::Shape::kN);

    typename Epilogue::OutputTileIterator iterator_D(
        gp.params_D, gp.ref_D.data(), gp.problem_size.mn(), thread_idx, threadblock_offset);

    typename Epilogue::OutputTileIterator it_dy(gp.params_D, const_cast<double *>(params.ptr_dy),
                                                gp.problem_size.mn(), thread_idx,
                                                threadblock_offset);
    typename Epilogue::OutputTileIterator it_dz(gp.params_D, const_cast<double *>(params.ptr_dz),
                                                gp.problem_size.mn(), thread_idx,
                                                threadblock_offset);
    typename Epilogue::OutputTileIterator it_ex(gp.params_D, const_cast<double *>(params.ptr_ex),
                                                gp.problem_size.mn(), thread_idx,
                                                threadblock_offset);
    typename Epilogue::OutputTileIterator it_ey(gp.params_D, const_cast<double *>(params.ptr_ey),
                                                gp.problem_size.mn(), thread_idx,
                                                threadblock_offset);
    typename Epilogue::OutputTileIterator it_ez(gp.params_D, const_cast<double *>(params.ptr_ez),
                                                gp.problem_size.mn(), thread_idx,
                                                threadblock_offset);

    AssemblyEpilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue.apply_assembly(iterator_D, accumulators, it_dy, it_dz, it_ex, it_ey, it_ez,
                            params.ptr_flux_bnd, params.ptr_lift1d,
                            params.ptr_lift_pair, params.Nq,
                            gp.problem_size.m(), params.nq_log2);
  }
};
