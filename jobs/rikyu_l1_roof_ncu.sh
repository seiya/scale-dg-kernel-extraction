#!/bin/bash
#SBATCH --job-name=dg_l1_roof_ncu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
# The GB200 half of the L1/shared third-term study: jobs/l1_roof_ncu.sh,
# unchanged, under Slurm.  Build for GB200 first
# (make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100), then
#   sbatch jobs/rikyu_l1_roof_ncu.sh
# 9 ncu invocations bounded at NCU_TIMEOUT (300 s) each, so a total hang ends
# at 45 min.  Job 82908 took one minute for all nine.
set -u
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/l1_roof_ncu.sh
