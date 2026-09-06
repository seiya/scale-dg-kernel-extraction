#!/bin/bash
#PJM -N dg_a100_p511_rev_ncu
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# The A100 half of the p=511 ranking-inversion study; see
# jobs/rikyu_p511_reversal_ncu.sh for the hypothesis and the metric choices.
# Fill in -g, build for cc80 as in jobs/wisteria_paths.sh, then submit FROM the
# checkout:
#   pjsub jobs/wisteria_p511_reversal_ncu.sh
#
# WALL CLOCK: TIME_BUDGET stops the sweep at 80 minutes, inside share-short's
# two-hour cap.  p=511 payload is 22.6 GiB against 40, so it fits.
set -u
export ORDERS="${ORDERS:-511}"
export PATHS="${PATHS:-CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM}"
export KERNELS="${KERNELS:-regex:tendency_|pdl_elembnd|cutlass|gemm|Kernel}"
export LAUNCH_SKIP="${LAUNCH_SKIP:-4}"
export LAUNCH_COUNT="${LAUNCH_COUNT:-4}"
export REPEATS="${REPEATS:-2}"
export CLOCKS="${CLOCKS:-none}"
export NSTEP="${NSTEP:-4}"
export NCU_TIMEOUT="${NCU_TIMEOUT:-900}"
export TIME_BUDGET="${TIME_BUDGET:-4800}"
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export OUTDIR="${OUTDIR:-a100_p511_reversal_ncu_${PJM_JOBID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/l1_roof_ncu.sh
