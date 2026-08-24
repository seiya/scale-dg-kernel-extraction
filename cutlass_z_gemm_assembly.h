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
// tile is staged through shared memory, the epilogue loads Dx, Dy, lift, and
// the three Escale fields and stores
//   dqdt = -(Ex*Dx + Ey*Dy + Ez*Dz + lift)
// instead of writing Dz.

template <typename Epilogue>
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
                      OutputTileIterator it_lift, OutputTileIterator it_ex,
                      OutputTileIterator it_ey, OutputTileIterator it_ez)
  {
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    SharedLoadIterator sli(this->shared_storage_.reference(), thread_idx_);

    #pragma unroll(1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      typename OutputTileIterator::Fragment frag_dx, frag_dy, frag_lift, frag_ex, frag_ey,
          frag_ez;
      it_dx.load(frag_dx);
      it_dy.load(frag_dy);
      it_lift.load(frag_lift);
      it_ex.load(frag_ex);
      it_ey.load(frag_ey);
      it_ez.load(frag_ez);
      ++it_dx;
      ++it_dy;
      ++it_lift;
      ++it_ex;
      ++it_ey;
      ++it_ez;

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
      auto const *lf = reinterpret_cast<double const *>(&frag_lift);
      auto const *ex = reinterpret_cast<double const *>(&frag_ex);
      auto const *ey = reinterpret_cast<double const *>(&frag_ey);
      auto const *ez = reinterpret_cast<double const *>(&frag_ez);
      auto *out = reinterpret_cast<double *>(&output_fragment);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < OutputTileIterator::Fragment::kElements; ++i) {
        out[i] = -(ex[i] * dx[i] + ey[i] * dy[i] + ez[i] * dz[i] + lf[i]);
      }

      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
  }
};

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_>
struct GemmBatchedDqdtAssembly {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using BaseKernel = cutlass::gemm::kernel::GemmBatched<Mma, Epilogue, ThreadblockSwizzle>;
  using AssemblyEpilogue = EpilogueDqdtAssembly<Epilogue>;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Params {
    typename BaseKernel::Params gemm{};
    double const *ptr_dx{nullptr};
    double const *ptr_dy{nullptr};
    double const *ptr_lift{nullptr};
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
      typename Epilogue::OutputTileIterator it_lift(gp.params_D,
                                                    const_cast<double *>(params.ptr_lift),
                                                    gp.problem_size.mn(), thread_idx,
                                                    threadblock_offset);
      it_lift.add_pointer_offset(gp.stride_D * batch_idx);
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
      epilogue.apply_assembly(iterator_D, accumulators, it_dx, it_dy, it_lift, it_ex, it_ey,
                              it_ez);
    }
  }
};
