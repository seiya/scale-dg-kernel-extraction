#!/bin/bash
#SBATCH --job-name=dg_roof_bytes
#SBATCH --gpus=1
#SBATCH --time=02:00:00
# GB200 half of jobs/roof_bytes_ncu.sh.  Build for GB200 first
#   make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100
# then  sbatch jobs/rikyu_roof_bytes_ncu.sh
# SHAPE stays 8x8x4: that is the GB200 default and 16x8x4 is free there
# (reports/h100_report.md 8.3), so the shape is not an axis on this machine.
set -u
export SHAPE="${SHAPE:-8x8x4}"
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/roof_bytes_ncu.sh
