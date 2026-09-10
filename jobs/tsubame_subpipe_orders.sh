#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dg_h100_subpipe_orders
#$ -j y
#
# H100 sub-pipe sweep over every order.  H100 stays the HOLD-OUT: these rows
# are for checking the roof identification and the eta transfer at the orders
# where reports/subpipe_transfer_model.md 5G says the model is out of its
# validity range, NOT for choosing anything in the model.
#
#   qsub -g <GROUP> jobs/tsubame_subpipe_orders.sh
#
# REQUIRES the current jobs/subpipe_ncu.sh (SHAPES knob + L1 / latency
# counters).  SHAPES is 8x8x4 to stay iso-shape with the calibration machines;
# add SHAPES="8x8x4 16x8x4" to get the production H100 configuration too.
set -u
export ORDERS="${ORDERS:-7 15 31 63 127 255 511 575}"
export PATHS="${PATHS:-CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM CUDAFORTRAN_FUSED}"
export SHAPES="${SHAPES:-8x8x4}"
export REPEATS="${REPEATS:-1}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export OUTDIR="${OUTDIR:-h100_subpipe_orders_${JOB_ID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
