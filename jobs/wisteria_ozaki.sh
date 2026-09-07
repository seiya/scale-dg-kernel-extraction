#!/bin/bash
#PJM -N dg_a100_ozaki
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# A100 Ozaki column -- measurement_inventory.md 19.7, third item.
# Same knobs as jobs/tsubame_ozaki.sh; the machine differences come from
# jobs/wisteria_paths.sh, which this reuses so that roofs, module names and
# the shape-spec exclusion stay in one place.
#
# BEFORE SUBMITTING (cc80 / sm_80):
#   module unload nvidia && module load nvidia/25.9   # do NOT load gcc/* or cuda/*
#   make clean
#   make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
#   fill in -g above, then:  pjsub jobs/wisteria_ozaki.sh
#
# 40 GiB is the binding constraint here, not time.  Ozaki carries slice and
# modulus workspaces on top of the fields, so the high orders may OOM where
# native GEMM fits.  That is a result; the body prints rc != 0 and continues.
# ORDERS_C is left at the wrapper's default (575) and p=767/1023 stay out.
set -u
P="CUDAFORTRAN_GEMM:CUDAFORTRAN_GEMM_OZAKI1:CUDAFORTRAN_GEMM_OZAKI2"
for p in 7 15 31 63 127 255 511 575; do export SCALE_DG_PATHS_$p="$P"; done
export ORDERS_A="7 15 31 63 127 255" ORDERS_B="511"
export ROUNDS=6 ROUNDS_HI=6 ROUNDS_XHI=3
export VALIDATE=0 TIMING=1
export OUTDIR="${OUTDIR:-a100_ozaki_${PJM_JOBID:-manual}}"
exec bash "${PJM_O_WORKDIR:-$PWD}/jobs/wisteria_paths.sh"
