#!/bin/bash
#PJM -N dg_a100_subpipe_orders
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# A100 sub-pipe sweep over every measurable order, so that the crossover map of
# reports/subpipe_transfer_model.md 5E.2 can be produced from EITHER calibration
# machine.  As it stands that map is a projection from GB200 alone; with this
# run the same map comes out of A100, and "the ranking flip is predicted from
# one machine's counters" stops depending on which machine was chosen.
#
# It also completes the source side for the two families that are still open:
# the FUSED (CUDA-core) kernels of section 5F.3, whose roof moves between
# machines, and the latency-bound family of section 5H.
#
# Fill in -g, build for cc80 as in jobs/wisteria_paths.sh, then submit FROM the
# checkout:
#   pjsub jobs/wisteria_subpipe_orders.sh
#   pjsub -x ORDERS=7:15:31 jobs/wisteria_subpipe_orders.sh   (override)
#
# TCS does not export the submitting shell either, and -x cannot carry a
# space, so lists go in colon-separated.
#
# REQUIRES the current jobs/subpipe_ncu.sh.  The copy that ran TCS 9568133
# predates both the SHAPES knob and the L1 / latency counters.
#
# WALL CLOCK: the same sweep took 11 min 27 s on GB200 (Slurm 92904) with
# REPEATS=1, and A100 runs these kernels 2-5x slower, so TIME_BUDGET stops it
# at 90 minutes inside share-short's two-hour cap.  p=575 payload is 32.9 GiB
# against 40, so it fits; p>=511 FUSED and p=575 FUSED / FUSED_TC do not exist
# and ERROR STOP at no cost.
set -u
export ORDERS="${ORDERS:-7 15 31 63 127 255 511 575}"
export PATHS="${PATHS:-CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM CUDAFORTRAN_FUSED}"
export SHAPES="${SHAPES:-8x8x4}"
export REPEATS="${REPEATS:-1}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export OUTDIR="${OUTDIR:-a100_subpipe_orders_${PJM_JOBID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
