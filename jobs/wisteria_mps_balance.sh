#!/bin/bash
#PJM -N dg_a100_mps_balance
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
# A100 half of the balance sweep.  See jobs/mps_balance.sh.
# 40% puts this machine at GB200's 25.2 DRAM bytes per SM-clock AND it already
# has GB200's fp64 tensor throughput per SM-clock, so that point differs from
# GB200 at 100% essentially only in the CUDA-core coordinate.
#   pjsub jobs/wisteria_mps_balance.sh          (fill in -g first)
set -u
export SCALE_DG_ROOT="${PJM_O_WORKDIR:-$PWD}"
export SCALE_DG_MODULES="nvidia/25.9" SCALE_DG_MODULE_PURGE=0
export ORDERS="${ORDERS:-63 255 511}" ROUNDS="${ROUNDS:-2}" TIME_BUDGET="${TIME_BUDGET:-5400}"
export OUTDIR="a100_mps_balance_${PJM_JOBID:-manual}"
exec bash "$SCALE_DG_ROOT/jobs/mps_balance.sh"
