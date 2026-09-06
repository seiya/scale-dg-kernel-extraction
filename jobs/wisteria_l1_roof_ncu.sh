#!/bin/bash
#PJM -N dg_a100_l1_roof_ncu
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# The A100 half of the L1/shared third-term study: jobs/l1_roof_ncu.sh,
# unchanged, under Fujitsu TCS.  Fill in -g, build for cc80 as in
# jobs/wisteria_paths.sh, then submit FROM the checkout:
#   pjsub jobs/wisteria_l1_roof_ncu.sh
#
# WALL CLOCK: share-short caps elapse at 02:00:00 (pjsub rejects more with
# SIM4720), so the job has to bound itself well inside that.  It does: nine ncu
# invocations at NCU_TIMEOUT=300 s each end at 45 minutes even if every one
# hangs.  The expected cost is far below that -- GB200 (Slurm 82908) ran all
# nine in one minute, about 7 s each -- and A100 replays more slowly but not by
# forty times.  If an invocation does time out, check the node with pjstat and
# resubmit excluding it before changing the profiler command (AGENTS.md).
set -u
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export OUTDIR="${OUTDIR:-a100_l1_roof_ncu_${PJM_JOBID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/l1_roof_ncu.sh
