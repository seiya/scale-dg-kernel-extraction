#!/bin/bash
#PJM -N dg_a100_subpipe_ncu
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# The A100 half of the SM sub-pipe study: jobs/subpipe_ncu.sh, unchanged, under
# Fujitsu TCS.  This is the second calibration machine, not a validation one --
# see the header of jobs/subpipe_ncu.sh for why GB200 alone cannot separate the
# fp64+tensor pipe from anything else proportional to SM-cycles.
#
# Fill in -g, build for cc80 as in jobs/wisteria_paths.sh, then submit FROM the
# checkout:
#   pjsub jobs/wisteria_subpipe_ncu.sh
#
# WALL CLOCK: share-short caps elapse at 02:00:00.  TIME_BUDGET stops the sweep
# at 90 minutes whatever happens.  p=511 payload is 22.6 GiB against 40, so it
# fits; p=511 CUDAFORTRAN_FUSED does not exist and ERROR STOPs at no cost.
#
# If an invocation times out, check the node with pjstat and resubmit excluding
# it before changing the profiler command (AGENTS.md).
set -u
export REPEATS="${REPEATS:-2}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export OUTDIR="${OUTDIR:-a100_subpipe_ncu_${PJM_JOBID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
