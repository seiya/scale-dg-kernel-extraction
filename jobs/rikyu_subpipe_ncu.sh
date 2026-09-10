#!/bin/bash
#SBATCH --job-name=dg_subpipe_ncu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
# The GB200 half of the SM sub-pipe study: jobs/subpipe_ncu.sh, unchanged,
# under Slurm.  Build for GB200 first
# (make CUDA=1 GPUFLAGS=-gpu=cc100), then
#   sbatch jobs/rikyu_subpipe_ncu.sh
#
# Twelve invocations (three orders x four paths); p=511 CUDAFORTRAN_FUSED does
# not exist and is expected to ERROR STOP immediately, which costs nothing.
# TIME_BUDGET bounds the sweep at 90 minutes inside the two-hour reservation.
set -u
export REPEATS="${REPEATS:-2}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
