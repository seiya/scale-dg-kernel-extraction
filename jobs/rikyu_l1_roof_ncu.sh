#!/bin/bash
#SBATCH --job-name=dg_l1_roof_ncu
#SBATCH --gpus=1
#SBATCH --time=03:00:00
# The GB200 half of the L1/shared third-term study: jobs/l1_roof_ncu.sh,
# unchanged, under Slurm.  Build for GB200 first
# (make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100), then
#   sbatch jobs/rikyu_l1_roof_ncu.sh
# 9 ncu invocations bounded at 900 s each, so a total hang ends at ~2.25 h.
set -u
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/l1_roof_ncu.sh
