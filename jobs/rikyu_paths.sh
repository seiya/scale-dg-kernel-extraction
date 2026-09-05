#!/bin/bash
#SBATCH --job-name=dg_paths_gb200
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#
# The GB200 half of the cross-path table: jobs/tsubame_paths.sh, unchanged,
# under Slurm instead of SGE.  Running the SAME script on both machines is what
# makes the H100/GB200 ratio one measurement rather than two conventions --
# same namelists, same interleaving, same timer, same median.
#
#   sbatch jobs/rikyu_paths.sh
#
# Build for GB200 first:
#   make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100
set -u
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/tsubame_paths.sh
