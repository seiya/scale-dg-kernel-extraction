#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dg_h100_subpipe_gaps
#$ -j y
#
# H100: the two gaps that only the target machine can close
# (reports/subpipe_transfer_model.md 5W.4 and 5V.1).  H100 stays the hold-out;
# neither phase feeds anything into the model.
#
# Phase 1 -- run-to-run spread.  Every sweep so far is REPEATS=1, and half the
# cells the model "misses" have true gaps under 2%, so whether those are model
# errors or measurement noise is not decidable yet.  The orders and paths are
# the ones the S/N < 1 cells sit on.
#
# Phase 2 -- the low-order unfused GEMM window.  Below p=63 one tendency call of
# GEMM / GEMM_CUTE does not fit in 24 launches: volume_flux and elembnd_flux
# never enter the window and the per-call total comes out ~42% low.  120
# launches covers it.
#
#   qsub -g <GROUP> jobs/tsubame_subpipe_gaps.sh
#
# Overrides go in colon-separated (-v ORDERS=63:511), never as ORDERS=... in
# front of qsub: SGE does not export the submitting shell.
set -u
ROOT="${SCALE_DG_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
export SCALE_DG_ROOT="$ROOT"
export SHAPES=8x8x4 CLOCKS=none NCU_TIMEOUT=1200

ORDERS="${ORDERS:-63 511 575 767}" \
PATHS="${PATHS:-CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_FUSED_TC}" \
REPEATS=3 TIME_BUDGET=4200 \
OUTDIR="h100_subpipe_repeat_${JOB_ID:-manual}" \
  bash jobs/subpipe_ncu.sh

ORDERS="7 15 31" PATHS="CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE" \
REPEATS=1 LAUNCH_SKIP=20 LAUNCH_COUNT=120 TIME_BUDGET=1800 \
OUTDIR="h100_subpipe_lowgemm_${JOB_ID:-manual}" \
  bash jobs/subpipe_ncu.sh
