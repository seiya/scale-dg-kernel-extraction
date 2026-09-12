#!/bin/bash
#PJM -N dg_a100_subpipe_rep
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# A100 column of the run-to-run width table of
# reports/subpipe_transfer_model.md 5Y.2.  Same body and same knobs as
# jobs/rikyu_subpipe_repeat.sh, which documents the two sets; that file is the
# one to read for why.
#
# High order (default, = TCS 9579033):
#   pjsub jobs/wisteria_subpipe_repeat.sh
#
# Low order, 5Y.5's hole.  TCS -x cannot carry a space and separates variables
# with commas, so the lists go in colon-separated (jobs/subpipe_ncu.sh splits
# on both):
#   pjsub -x ORDERS=7:15:31,PATHS=CUDAFORTRAN_FUSED:CUDAFORTRAN_FUSED_TC:CUDAFORTRAN_FUSED_DFMA:CUDAFORTRAN_GEMM:CUDAFORTRAN_GEMM_CUTE:CUDAFORTRAN_GEMM_FUSED,NCU_TIMEOUT=600 \
#         jobs/wisteria_subpipe_repeat.sh
#
# A100 runs these kernels 2-5x slower than GB200, so TIME_BUDGET stops the
# sweep inside share-short's two-hour cap.  Fill in -g and build for cc80 as in
# jobs/wisteria_paths.sh first.
set -u
export ORDERS="${ORDERS:-63 511 575}"
export PATHS="${PATHS:-CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_FUSED_TC}"
export SHAPES="${SHAPES:-8x8x4}"
export REPEATS="${REPEATS:-3}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export NCU_TIMEOUT="${NCU_TIMEOUT:-1500}"
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export OUTDIR="${OUTDIR:-a100_subpipe_repeat_${PJM_JOBID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
