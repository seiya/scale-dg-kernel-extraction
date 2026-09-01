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

#include "cutlass_z_gemm_assembly.h"   // RepadEpilogue

//- y volume GEMM carrying the final assembly epilogue, i.e. the same fusion
//- package GEMM_FUSED normally puts on z, moved onto y.  AGENTS.md leaves the
//- choice of carrier open: it only requires that the LAST volume GEMM fuse the
//- weighting and the surface lift, and that the library assignment and the
//- mainloop tiles stay the same as GEMM_CUTE's.  This header changes neither:
//- the mainloop is Set::GemmY / Set::GemmY32, exactly what GEMM_CUTE runs.
//
//- Geometry.  The y GEMM is batched over b = k + Nq*elem with a per-batch
//- problem (m = Nq, n = Nq, k = Nq) and a column-major C, which CUTLASS solves
//- transposed.  So an epilogue tile row is the problem's n index = j and an
//- epilogue tile column is the problem's m index = i; the z index k and the
//- element are fixed for the whole tile.
//
//- What that buys, against the z carrier (rows = k, columns = p = i + j*Nq):
//-   z carrier: lz is separable (coefficient per row, face value per column),
//-              lx and ly each need a per-element face load -> 4 loads/element.
//-   y carrier: lx is separable (coefficient per column i, face value per row
//-              j at fixed k) and ly is separable the other way, so only lz
//-              needs a per-element face load -> 2 loads/element, and its two
//-              coefficients are loop-invariant scalars of the batch.
//- The volume-tensor count is unchanged: five (Dx, Dz, Ex, Ey, Ez), or four
//- when the x GEMM already applied Escale_x (kXWeighted).  Escale_y cannot be
//- forwarded here -- it multiplies this kernel's own accumulator.
template <typename Epilogue, bool kXWeighted = false>
class EpilogueDqdtAssemblyY : public Epilogue {
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
  EpilogueDqdtAssemblyY(SharedStorage &shared_storage, int thread_idx, int warp_idx,
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
                      OutputTileIterator it_dx, OutputTileIterator it_dz,
                      OutputTileIterator it_ex, OutputTileIterator it_ey,
                      OutputTileIterator it_ez, double const *flux_bnd,
                      double const *lift1d, int Nq, int kz)
  {
    using ThreadMap = typename OutputTileIterator::ThreadMap;
    static int const kEPV = ThreadMap::kElementsPerAccess;
    static int const kCols = ThreadMap::Iterations::kColumn * kEPV;

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    SharedLoadIterator sli(this->shared_storage_.reference(), thread_idx_);

    const int nq2 = Nq * Nq;
    double const *fb0 = flux_bnd;
    double const *fb1 = flux_bnd + nq2;
    double const *fb2 = flux_bnd + 2 * nq2;
    double const *fb3 = flux_bnd + 3 * nq2;
    double const *fb4 = flux_bnd + 4 * nq2;
    double const *fb5 = flux_bnd + 5 * nq2;

    //- k is fixed for the batch, so the two z-face lift coefficients are two
    //- scalars for the whole tile.  Their face values are the one term that
    //- still has to be gathered per output element.
    const double lz1 = lift1d[4 * Nq + kz];
    const double lz2 = lift1d[5 * Nq + kz];
    const int kNq = kz * Nq;

    //- Column-invariant part: i, the two x-face lift coefficients (which
    //- depend on i) and the two y-face VALUES (which depend on i and k).
    int col_i[kCols];
    double col_lx1[kCols];
    double col_lx2[kCols];
    double col_fy0[kCols];
    double col_fy2[kCols];
    {
      const int start_col = destination_iterator.thread_start_column();
      CUTLASS_PRAGMA_UNROLL
      for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kEPV; ++e) {
          const int cc = column * kEPV + e;
          const int c = start_col + column * ThreadMap::Delta::kColumn + e;
          const int i = min(c, Nq - 1);
          col_i[cc] = i;
          col_lx1[cc] = lift1d[Nq + i];
          col_lx2[cc] = lift1d[3 * Nq + i];
          col_fy0[cc] = fb0[i + kNq];
          col_fy2[cc] = fb2[i + kNq];
        }
      }
    }

    #pragma unroll(1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      typename OutputTileIterator::Fragment frag_dx, frag_dz, frag_ex, frag_ey, frag_ez;
      if (kXWeighted) {
        //- Escale_x is already in deriv_x.
        it_dx.load(frag_dx);
        it_dz.load(frag_dz);
        it_ey.load(frag_ey);
        it_ez.load(frag_ez);
        ++it_dx;
        ++it_dz;
        ++it_ey;
        ++it_ez;
      } else {
        it_dx.load(frag_dx);
        it_dz.load(frag_dz);
        it_ex.load(frag_ex);
        it_ey.load(frag_ey);
        it_ez.load(frag_ez);
        ++it_dx;
        ++it_dz;
        ++it_ex;
        ++it_ey;
        ++it_ez;
      }

      // lift(i,j,k), summed in the same (x+y)+z order the two-kernel form used.
      double frag_lift[OutputTileIterator::Fragment::kElements];
      const int start_row = destination_iterator.thread_start_row();
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
            const int j = min(start_row + row_offset, Nq - 1);
            //- Row-invariant part: the two y-face lift coefficients (which
            //- depend on j) and the two x-face VALUES (which depend on j, k).
            const double ly1 = lift1d[j];
            const double ly2 = lift1d[2 * Nq + j];
            const double fx1 = fb1[j + kNq];
            const double fx3 = fb3[j + kNq];
            const int jNq = j * Nq;

            CUTLASS_PRAGMA_UNROLL
            for (int cc = 0; cc < kCols; ++cc) {
              const int idx = frag_row_idx * kCols + cc;
              const int p = col_i[cc] + jNq;
              const double lx = col_lx1[cc] * fx1 + col_lx2[cc] * fx3;
              const double ly = ly1 * col_fy0[cc] + ly2 * col_fy2[cc];
              const double lz = lz1 * fb4[p] + lz2 * fb5[p];
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

      typename OutputTileIterator::Fragment output_fragment;
      auto const *dy = reinterpret_cast<double const *>(&aligned_accum_fragment[0]);
      auto const *dx = reinterpret_cast<double const *>(&frag_dx);
      auto const *dz = reinterpret_cast<double const *>(&frag_dz);
      auto const *lf = frag_lift;
      auto const *ex = reinterpret_cast<double const *>(&frag_ex);
      auto const *ey = reinterpret_cast<double const *>(&frag_ey);
      auto const *ez = reinterpret_cast<double const *>(&frag_ez);
      auto *out = reinterpret_cast<double *>(&output_fragment);
      if (kXWeighted) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < OutputTileIterator::Fragment::kElements; ++i) {
          out[i] = -(dx[i] + ey[i] * dy[i] + ez[i] * dz[i] + lf[i]);
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

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_,
          bool kXWeighted = false>
struct GemmBatchedDqdtAssemblyY {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using BaseKernel = cutlass::gemm::kernel::GemmBatched<Mma, Epilogue, ThreadblockSwizzle>;
  using AssemblyEpilogue = EpilogueDqdtAssemblyY<Epilogue, kXWeighted>;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Params {
    typename BaseKernel::Params gemm{};
    double const *ptr_dx{nullptr};
    double const *ptr_dz{nullptr};
    double const *ptr_flux_bnd{nullptr};
    double const *ptr_lift1d{nullptr};
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
      typename Epilogue::OutputTileIterator it_dz(gp.params_D, const_cast<double *>(params.ptr_dz),
                                                  gp.problem_size.mn(), thread_idx,
                                                  threadblock_offset);
      it_dz.add_pointer_offset(gp.stride_D * batch_idx);
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
      //- batch b = k + Nq*elem: the element selects the face buffer, k selects
      //- the z-face lift coefficients and the row/column face slices.
      const int nfp_tot = 6 * params.Nq * params.Nq;
      const int elem = batch_idx / params.Nq;
      const int kz = batch_idx - elem * params.Nq;
      epilogue.apply_assembly(iterator_D, accumulators, it_dx, it_dz, it_ex, it_ey, it_ez,
                              params.ptr_flux_bnd + int64_t(nfp_tot) * elem,
                              params.ptr_lift1d, params.Nq, kz);
    }
  }
};
