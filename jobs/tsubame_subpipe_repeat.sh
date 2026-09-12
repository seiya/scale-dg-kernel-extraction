#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dg_h100_subpipe_rep
#$ -j y
#
# H100 column of the run-to-run width table of
# reports/subpipe_transfer_model.md 5Y.2.  Same body and same knobs as
# jobs/rikyu_subpipe_repeat.sh, which documents the two sets.
#
# The published H100 width is TWO rounds (SGE 8631255 ran out of TIME_BUDGET
# before the third), so the high-order set is worth re-running as well as the
# low-order one: a range over n=2 is not a width.
#
# High order (default):
#   qsub -g <GROUP> jobs/tsubame_subpipe_repeat.sh
#
# Low order, 5Y.5's hole.  SGE -v cannot carry a space, so the lists go in
# colon-separated (jobs/subpipe_ncu.sh splits on colons and commas), and
# VAR=... in front of qsub does nothing:
#   qsub -g <GROUP> -v ORDERS=7:15:31,NCU_TIMEOUT=600,\
#   PATHS=CUDAFORTRAN_FUSED:CUDAFORTRAN_FUSED_TC:CUDAFORTRAN_FUSED_DFMA:CUDAFORTRAN_GEMM:CUDAFORTRAN_GEMM_CUTE:CUDAFORTRAN_GEMM_FUSED \
#        jobs/tsubame_subpipe_repeat.sh
#
# H100 is the hold-out machine: these rows say how big its noise is, they do
# not calibrate anything.
set -u
export ORDERS="${ORDERS:-63 511 575}"
export PATHS="${PATHS:-CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_FUSED_TC}"
export SHAPES="${SHAPES:-8x8x4}"
export REPEATS="${REPEATS:-3}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-6000}"
export NCU_TIMEOUT="${NCU_TIMEOUT:-1500}"
export OUTDIR="${OUTDIR:-h100_subpipe_repeat_${JOB_ID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
