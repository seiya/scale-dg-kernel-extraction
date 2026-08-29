#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/kernel/gemm_batched.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/array.h"

// y-volume GEMM with the standard TensorOp mainloop and an epilogue that takes
// two source tensors instead of one:
//
//   deriv_xy = Escale_y * acc + deriv_x
//
// kMulAddend (Nq <= 64) multiplies the addend by Escale_x as well:
//   deriv_xy = Escale_y * acc + Escale_x * deriv_x
// because that branch's x GEMM is cuBLAS and cannot weight Escale_x itself.
// The Nq > 64 branch leaves kMulAddend false: deriv_x already holds Ex*Dx.
//
// The point is not the arithmetic -- the same two multiplies and one add used
// to happen, split between this epilogue (Escale_y) and the z epilogue (the
// dx + dy sum). The point is which kernel reads deriv_x.
//
// The z GEMM is the only memory-limited kernel of the three: at p=127 it reads
// flux_z, deriv_x, deriv_y and Escale_z and writes dqdt, five streams, and
// making any one of the 134 MB streams L2-resident is worth 63-72 us of its
// 211 us (ablation, reports/p127_gap_study.md). The x and y GEMMs are issue
// bound on the DMMA pipe at 79-80% with DRAM at 18%, so a stream costs them
// much less. Folding the dx + dy sum in here moves one read out of the kernel
// that cannot absorb it into one that can, and deriv_x is still warm in L2
// because the x GEMM wrote it one kernel ago.
//
// The sum order is unchanged: dqdt = -((dx + dy) + Ez*Dz + lift), and floating
// point addition is commutative, so dy + dx here gives the same bits.
template <typename Epilogue, bool kMulAddend>
class EpilogueScaleAdd : public Epilogue {
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
  EpilogueScaleAdd(SharedStorage &shared_storage, int thread_idx, int warp_idx, int lane_idx)
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
  void apply_scale_add(OutputTileIterator destination_iterator,
                       AccumulatorTile const &accumulators, OutputTileIterator it_scale,
                       OutputTileIterator it_add, OutputTileIterator it_mul)
  {
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    SharedLoadIterator sli(this->shared_storage_.reference(), thread_idx_);

    #pragma unroll(1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      typename OutputTileIterator::Fragment frag_scale, frag_add, frag_mul;
      it_scale.load(frag_scale);
      it_add.load(frag_add);
      ++it_scale;
      ++it_add;
      if constexpr (kMulAddend) {
        it_mul.load(frag_mul);
        ++it_mul;
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
      auto const *acc = reinterpret_cast<double const *>(&aligned_accum_fragment[0]);
      auto const *sc = reinterpret_cast<double const *>(&frag_scale);
      auto const *ad = reinterpret_cast<double const *>(&frag_add);
      auto const *mu = reinterpret_cast<double const *>(&frag_mul);
      auto *out = reinterpret_cast<double *>(&output_fragment);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < OutputTileIterator::Fragment::kElements; ++i) {
        if constexpr (kMulAddend) {
          out[i] = ad[i] * mu[i] + acc[i] * sc[i];
        } else {
          out[i] = ad[i] + acc[i] * sc[i];
        }
      }

      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
  }
};

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_,
          bool kMulAddend>
struct GemmBatchedScaleAdd {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using BaseKernel = cutlass::gemm::kernel::GemmBatched<Mma, Epilogue, ThreadblockSwizzle>;
  using ScaleAddEpilogue = EpilogueScaleAdd<Epilogue, kMulAddend>;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Params {
    typename BaseKernel::Params gemm{};
    double const *ptr_scale{nullptr};
    double const *ptr_add{nullptr};
    double const *ptr_mul{nullptr};
    long long stride_scale{0};
    long long stride_add{0};
    long long stride_mul{0};
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

      typename Epilogue::OutputTileIterator it_scale(
          gp.params_D, const_cast<double *>(params.ptr_scale), gp.problem_size.mn(), thread_idx,
          threadblock_offset);
      it_scale.add_pointer_offset(params.stride_scale * batch_idx);

      typename Epilogue::OutputTileIterator it_add(
          gp.params_D, const_cast<double *>(params.ptr_add), gp.problem_size.mn(), thread_idx,
          threadblock_offset);
      it_add.add_pointer_offset(params.stride_add * batch_idx);

      typename Epilogue::OutputTileIterator it_mul(
          gp.params_D,
          const_cast<double *>(kMulAddend ? params.ptr_mul : params.ptr_add),
          gp.problem_size.mn(), thread_idx, threadblock_offset);
      if constexpr (kMulAddend) {
        it_mul.add_pointer_offset(params.stride_mul * batch_idx);
      }

      ScaleAddEpilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
      epilogue.apply_scale_add(iterator_D, accumulators, it_scale, it_add, it_mul);
    }
  }
};
