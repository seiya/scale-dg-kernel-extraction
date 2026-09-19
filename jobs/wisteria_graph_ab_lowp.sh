#!/bin/bash
#PJM -N dg_a100_graph_ab
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
# Wisteria/BDEC-01 Aquarius (A100 40 GiB) wrapper for jobs/graph_ab_lowp.sh
# (CUDA-graph on/off A/B, p=7 and 15).  Build with nvidia/25.9, cc80 / sm_80
# (see jobs/wisteria_paths.sh for the module and host-compiler notes), fill in
# -g, then
#   pjsub jobs/wisteria_graph_ab_lowp.sh
# No sm_90 MMA shapes on Ampere: the six paper paths at 8x8x4 only.
set -u
export SCALE_DG_MACHINE=a100
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export SCALE_DG_RESOURCE="pjsub gpu=1 (A100 40GiB)"
export SCALE_DG_SHAPE_SPECS=
export OUTDIR="${OUTDIR:-graph_ab_lowp_${PJM_JOBID:-manual}}"
export SCALE_DG_ROOT=${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/graph_ab_lowp.sh
