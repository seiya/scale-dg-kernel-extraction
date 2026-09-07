#!/bin/bash
#PJM -N dg_a100_acc
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# A100 OpenACC column -- measurement_inventory.md 19.7, third item.
# Needs its own build, in a worktree, exactly as jobs/tsubame_acc.sh explains:
#
#   git worktree add --detach ../dg-acc HEAD
#   cd ../dg-acc && ln -s <main>/third_party third_party
#   module unload nvidia && module load nvidia/25.9
#   make ACC=1 GPUFLAGS=-gpu=cc80
#   fill in -g, then:  pjsub -x EXE=$PWD/scale-dg_extraction <main>/jobs/wisteria_acc.sh
set -u
: "${EXE:?set EXE to the OpenACC executable, e.g. -x EXE=/path/scale-dg_extraction}"
export EXE
for p in 7 15; do export SCALE_DG_PATHS_$p="OPENACC_ASIS:OPENACC_SPLIT"; done
export ORDERS_A="7 15" ORDERS_B="" ORDERS_C=""
export ROUNDS=6 VALIDATE=0 TIMING=1
export OUTDIR="${OUTDIR:-a100_acc_${PJM_JOBID:-manual}}"
exec bash "${PJM_O_WORKDIR:-$PWD}/jobs/wisteria_paths.sh"
