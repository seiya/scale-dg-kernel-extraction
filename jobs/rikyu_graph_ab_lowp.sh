#!/bin/bash
#SBATCH --job-name=graph_ab_lowp
#SBATCH --gpus=1
#SBATCH --time=00:40:00
# GB200 wrapper for jobs/graph_ab_lowp.sh (CUDA-graph on/off A/B, p=7 and 15).
#   make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100
#   sbatch jobs/rikyu_graph_ab_lowp.sh
# GB200 expands 16x8x4 into DMMA.8x8x4 (+0.03%), so the preferred-shape specs
# are dropped here: the six paper paths at the tree's default shape.
set -u
export SCALE_DG_MACHINE=gb200
export SCALE_DG_RESOURCE="sbatch --gpus=1 (GB200 1 GPU)"
export SCALE_DG_SHAPE_SPECS=
export SCALE_DG_ROOT=${SCALE_DG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/graph_ab_lowp.sh
