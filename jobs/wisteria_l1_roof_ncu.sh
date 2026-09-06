#!/bin/bash
#PJM -N dg_a100_l1_roof_ncu
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=2:30:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# The A100 half of the L1/shared third-term study: jobs/l1_roof_ncu.sh,
# unchanged, under Fujitsu TCS.  Fill in -g, build for cc80 as in
# jobs/wisteria_paths.sh, then submit FROM the checkout:
#   pjsub jobs/wisteria_l1_roof_ncu.sh
#
# WALL CLOCK: 9 ncu invocations bounded at 900 s each, so even a total hang
# ends at about 2.25 h, inside the 2.5 h asked for.  ncu on A100 is expected
# to be slower per replay than on GB200; if an invocation times out, check
# the node with pjstat and resubmit elsewhere before changing the command
# (AGENTS.md).
set -u
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export OUTDIR="${OUTDIR:-a100_l1_roof_ncu_${PJM_JOBID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/l1_roof_ncu.sh
