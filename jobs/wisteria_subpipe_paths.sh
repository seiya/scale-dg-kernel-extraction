#!/bin/bash
#PJM -N dg_a100_subpipe_paths
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# A100 sub-pipe sweep over the three paths the model has never been
# transferred for: FUSED_DFMA, GEMM_CUTE and OZAKI1.  Same reason as
# jobs/tsubame_subpipe_paths.sh -- FUSED_DFMA vs FUSED_TC is the mma
# instruction axis and GEMM_CUTE vs GEMM_FUSED the fusion axis, so both are
# rankings the model should predict, and neither needs a new machine.
#
# Fill in -g, build for cc80 as in jobs/wisteria_paths.sh, then submit FROM the
# checkout:
#   pjsub jobs/wisteria_subpipe_paths.sh
#
# REQUIRES the current jobs/subpipe_ncu.sh.  The copy that ran TCS 9568133
# predates both the SHAPES knob and the L1 / latency counters.
#
# WALL CLOCK: the same sweep took 11 min 27 s on GB200 (Slurm 92904) with
# REPEATS=1, and A100 runs these kernels 2-5x slower, so TIME_BUDGET stops it
# at 90 minutes inside share-short's two-hour cap.  p=575 payload is 32.9 GiB
# against 40, so it fits; p>=511 FUSED and p=575 FUSED / FUSED_TC do not exist
# and ERROR STOP at no cost.
set -u
export ORDERS="${ORDERS:-7 15 31 63 127 255 511 575}"
export PATHS="${PATHS:-CUDAFORTRAN_FUSED_DFMA CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_OZAKI1}"
export SHAPES="${SHAPES:-8x8x4}"
export REPEATS="${REPEATS:-1}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export OUTDIR="${OUTDIR:-a100_subpipe_paths_${PJM_JOBID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
