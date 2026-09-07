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
# EXE must be an ABSOLUTE path to the OpenACC executable.  Do not pass $PWD
# through the scheduler: job 9539009 was submitted that way, the variable
# arrived empty, EXE became "/scale-dg_extraction", and all 24 runs failed
# with "No such file or directory".  Write the path out, and this checks it.
if [ -z "${EXE:-}" ]; then
  echo "ERROR: set EXE to the OpenACC executable (absolute path)." >&2
  echo "   e.g.  pjsub -x EXE=/work/.../dg-acc/scale-dg_extraction ..." >&2
  exit 2
fi
case "$EXE" in /*) ;; *) echo "ERROR: EXE must be absolute, got '$EXE'" >&2; exit 2;; esac
[ -x "$EXE" ] || { echo "ERROR: EXE is not an executable file: '$EXE'" >&2; exit 2; }
echo "EXE=$EXE"
export EXE
for p in 7 15; do export SCALE_DG_PATHS_$p="OPENACC_ASIS:OPENACC_SPLIT"; done
export ORDERS_A="7 15" ORDERS_B="" ORDERS_C=""
export ROUNDS=6 VALIDATE=0 TIMING=1
export OUTDIR="${OUTDIR:-a100_acc_${PJM_JOBID:-manual}}"
exec bash "${PJM_O_WORKDIR:-$PWD}/jobs/wisteria_paths.sh"
