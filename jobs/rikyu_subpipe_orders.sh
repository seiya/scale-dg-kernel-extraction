#!/bin/bash
#SBATCH --job-name=dg_subpipe_orders
#SBATCH --gpus=1
#SBATCH --time=03:00:00
# GB200 sub-pipe sweep over EVERY measurable order.
#
# reports/subpipe_transfer_model.md is built on p=63 / 255 / 511.  The claim the
# paper needs is stronger: that the model says WHERE the ranking flips, which
# means it has to run over the whole order axis, including the low orders where
# section 5D shows it is out of its validity range.  The target-machine
# timings already exist for every order (the roof_bytes sweeps on all three
# machines), so only the SOURCE machine needs the sub-pipe counters.
#
# Also adds the L1 sector/request/wavefront counters: section 6 lists the L1
# capacity as the weakest number in the model, and it stays fitted until the
# global path through L1TEX is counted rather than inferred from a max-over-
# sub-pipes percentage.
#
# p=575 has no FUSED_TC / FUSED and p>=511 no FUSED; those cells ERROR STOP at
# no cost.  REPEATS=1 because job 92507 already measured the round-to-round
# spread at three of these orders with REPEATS=2.
set -u
export ORDERS="${ORDERS:-7 15 31 63 127 255 511 575}"
export PATHS="${PATHS:-CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM CUDAFORTRAN_FUSED}"
export SHAPES="${SHAPES:-8x8x4}"
export REPEATS="${REPEATS:-1}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-9000}"
export OUTDIR="${OUTDIR:-subpipe_orders_${SLURM_JOB_ID:-manual}}"
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
