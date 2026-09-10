#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dg_h100_subpipe_paths
#$ -j y
#
# H100 sub-pipe sweep over the three paths the model has never been transferred
# for: FUSED_DFMA, GEMM_CUTE and OZAKI1.  H100 stays the HOLD-OUT.
#
# Why these three: FUSED_DFMA is the iso-schedule ablation of the mma
# instruction against FUSED_TC, and GEMM_CUTE the unfused control against
# GEMM_FUSED, so each is a one-axis pair whose RANKING the model should get for
# the same reason it gets GEMM_FUSED vs FUSED_TC.  They widen the denominator of
# the 21/21 in reports/subpipe_transfer_model.md 5L.4 without adding any new
# machine.
#
#   qsub -g <GROUP> jobs/tsubame_subpipe_paths.sh
#
# SGE does not export the submitting shell, and -v cannot carry a space, so
# overrides go in as -v ORDERS=7:15:31 (colons; jobs/subpipe_ncu.sh splits).
#
# REQUIRES the current jobs/subpipe_ncu.sh (SHAPES knob + L1 / latency
# counters).  SHAPES is 8x8x4 to stay iso-shape with the calibration machines;
# add SHAPES="8x8x4 16x8x4" to get the production H100 configuration too.
set -u
export ORDERS="${ORDERS:-7 15 31 63 127 255 511 575}"
export PATHS="${PATHS:-CUDAFORTRAN_FUSED_DFMA CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_OZAKI1}"
export SHAPES="${SHAPES:-8x8x4}"
export REPEATS="${REPEATS:-1}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export OUTDIR="${OUTDIR:-h100_subpipe_paths_${JOB_ID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
