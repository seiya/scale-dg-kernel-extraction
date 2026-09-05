#!/bin/bash
#PJM -N dg_a100_cublas_names
#PJM -L rscgrp=CHANGE_ME
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# Prediction 3 of reports/a100_prediction.md, on A100: does cuBLAS dispatch
# cutlass_80_tensorop_d884gemm for the FP64 volume GEMMs on the architecture
# that kernel was generated for?  jobs/cublas_kernel_names_nsys.sh, unchanged,
# under Fujitsu TCS.
#
# Fill in the two CHANGE_ME (resource group, project group), build for cc80 as
# in jobs/wisteria_paths.sh, then submit FROM the checkout:
#   pjsub jobs/wisteria_cublas_nsys.sh
#
# Short job: 2 orders x 2 paths x nstep=10 under nsys.  An hour is generous;
# if it times out, check `pjstat` for the node and resubmit excluding it before
# touching the profiler command (AGENTS.md).
set -u

export SCALE_DG_MODULES="nvidia/25.9 gcc/12.2.0"
export SCALE_DG_MODULE_PURGE=0
export OUTDIR="${OUTDIR:-a100_cublas_names_${PJM_JOBID:-manual}}"

ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/cublas_kernel_names_nsys.sh
