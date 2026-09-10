#!/bin/bash
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=0:20:00
#PJM -N dg_a100_clat
#PJM -j
# A100 (Wisteria/BDEC-01 Aquarius).  nvidia/* conflicts with gcc/*; do not load
# both (reports/a100_prediction.md 8.1).
export SCALE_DG_ROOT="${PJM_O_WORKDIR:-$PWD}"
export SCALE_DG_MODULES="nvidia/25.9"
export ARCH=sm_80
export OUT="contended_latency_sm80_${PJM_JOBID:-manual}"
exec bash "$SCALE_DG_ROOT/jobs/contended_latency_probe.sh"
