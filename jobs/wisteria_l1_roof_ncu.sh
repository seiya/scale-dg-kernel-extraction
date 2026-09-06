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
# SIM4720), so the job has to bound itself well inside that.  It does, but no
# longer through the per-invocation timeout: this is the retake of
# reports/a100_report.md 11.7, two clock-control modes x ten interleaved
# repeats x nine configurations, so 180 invocations rather than nine.
# TIME_BUDGET stops the sweep at 90 minutes whatever happens, leaving half an
# hour of headroom inside the cap.  The expected cost is far below it: job
# 9523261 ran nine invocations in 1 min 49 s, about 12 s each, so 180 is
# roughly 36 minutes.
#
# Set REPEATS=1 CLOCKS=base to reproduce job 9523261 exactly.
#
# If an invocation times out, check the node with pjstat and resubmit
# excluding it before changing the profiler command (AGENTS.md).
set -u
export REPEATS="${REPEATS:-10}"
export CLOCKS="${CLOCKS:-base none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export OUTDIR="${OUTDIR:-a100_l1_roof_ncu_${PJM_JOBID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/l1_roof_ncu.sh
