#!/bin/bash
#PJM -N dg_a100_subpipe_gaps
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# A100: the same two gaps as jobs/tsubame_subpipe_gaps.sh -- the run-to-run
# spread (5W.4) and the low-order unfused GEMM window (5V.1).  A100 stays a
# calibration machine; neither phase changes the model, they bound the
# measurement.
#
# p=767 is absent because A100 cannot hold it (76 GiB against 40).  p=575 is
# 32.9 GiB and fits; p>=511 FUSED_TC does not exist and ERROR STOPs at no cost.
#
#   pjsub jobs/wisteria_subpipe_gaps.sh        (fill in -g first)
#   pjsub -x ORDERS=63:511 ...                 (overrides colon-separated)
set -u
ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
export SCALE_DG_ROOT="$ROOT"
export SCALE_DG_MODULES="nvidia/25.9" SCALE_DG_MODULE_PURGE=0
export SHAPES=8x8x4 CLOCKS=none NCU_TIMEOUT=1200

ORDERS="${ORDERS:-63 511 575}" \
PATHS="${PATHS:-CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_FUSED_TC}" \
REPEATS=3 TIME_BUDGET=4200 \
OUTDIR="a100_subpipe_repeat_${PJM_JOBID:-manual}" \
  bash jobs/subpipe_ncu.sh

ORDERS="7 15 31" PATHS="CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE" \
REPEATS=1 LAUNCH_SKIP=20 LAUNCH_COUNT=120 TIME_BUDGET=1800 \
OUTDIR="a100_subpipe_lowgemm_${PJM_JOBID:-manual}" \
  bash jobs/subpipe_ncu.sh
