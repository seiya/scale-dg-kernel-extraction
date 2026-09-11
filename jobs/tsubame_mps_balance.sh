#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dg_h100_mps_balance
#$ -j y
# H100 half of the balance sweep.  See jobs/mps_balance.sh for what it measures.
# 36% puts this machine at GB200's 25.2 DRAM bytes per SM-clock.
#   qsub -g <GROUP> jobs/tsubame_mps_balance.sh
# Overrides colon-separated: -v PCTS=100:50:36, ORDERS=63:511
set -u
export SCALE_DG_ROOT="${SGE_O_WORKDIR:-$PWD}"
export SCALE_DG_MODULES="nvhpc"
export ORDERS="${ORDERS:-63 255 511}" ROUNDS="${ROUNDS:-2}" TIME_BUDGET="${TIME_BUDGET:-5400}"
export OUTDIR="h100_mps_balance_${JOB_ID:-manual}"
exec bash "$SCALE_DG_ROOT/jobs/mps_balance.sh"
