#!/bin/bash
#PJM -N dg_a100_mps_match
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
# The matched-balance point of reports/subpipe_transfer_model.md 5Z, taken for
# EVERY cell the balance sweep has, not just the one ratio.
#
# GB200 at 100% is 25.21 DRAM bytes per SM-clock.  The balance sweep
# (jobs/wisteria_mps_balance.sh, TCS 9580356) ran A100 at 40%, which the smid
# probe showed is 42 SMs = 26.25 B/SM-clk, 4.1% leaner than the balance it was
# meant to match; the matched point is 41% = 44 SMs = 25.06 (-0.6%).  The
# first version of this job (TCS 9580806) took 41% for p=511 and the two paths
# of the main ratio only, so every other order and path of the "matched
# balance" comparison was still standing on the 42-SM data.  This run takes
# 41% for the sweep's full grid -- p=63/255/511, all six paths, three rounds --
# with 40% interleaved in the same job so the 42-vs-44 SM difference is an
# in-job A/B and not a comparison across two jobs.  The SM count granted at
# each percentage is measured inside the job (smid.txt), never borrowed.
#
# Same body, same timers, same namelists as the sweep, so the 41% column drops
# into analysis/subpipe_balance_times.csv beside the others.
#   pjsub jobs/wisteria_mps_match.sh          (fill in -g first)
# The same body re-takes the sweep's 100% calibration point, whose p=63
# FUSED_TC / FUSED rows in 9580356 are the cold-start cells of 5AG.3.4:
#   pjsub -x TAG=full,PCTS=100,ROUNDS=5 jobs/wisteria_mps_match.sh
set -u
export SCALE_DG_ROOT="${PJM_O_WORKDIR:-$PWD}"
export SCALE_DG_MODULES="nvidia/25.9" SCALE_DG_MODULE_PURGE=0
export ORDERS="${ORDERS:-63 255 511}" PCTS="${PCTS:-41 40}" ROUNDS="${ROUNDS:-3}"
export TIME_BUDGET="${TIME_BUDGET:-3000}"
export OUTDIR="a100_mps_${TAG:-match}_${PJM_JOBID:-manual}"
exec bash "$SCALE_DG_ROOT/jobs/mps_balance.sh"
