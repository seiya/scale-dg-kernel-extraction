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

# The shared body defaults every one of these to H100, because TSUBAME was the
# first machine to run it.  Without the overrides a GB200 run writes
# machine=h100 and the H100 roofs into its own metadata.txt, which is the file
# reports cite so they never have to guess which peaks a run was scored
# against.  wisteria_paths.sh overrides all four for A100; this wrapper did
# not, and job 81583 (2026-09-05, the nine-order retake) carries the wrong
# four.
#
# GB200: 40.1 TF FP64 on the CUDA cores and the SAME 40.1 on the Tensor Cores
# -- the 2x that Ampere and Hopper have is gone here, so a path that issues
# mma is scored against the same denominator as one that does not.  7.9 TB/s
# HBM3e.
export SCALE_DG_MACHINE=gb200
export SCALE_DG_PEAK_FP64_CUDA=40.1
export SCALE_DG_PEAK_FP64_TENSOR=40.1
export SCALE_DG_PEAK_HBM=7.9
export SCALE_DG_RESOURCE="sbatch --gpus=1 (GB200 1 GPU)"

export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/tsubame_paths.sh
