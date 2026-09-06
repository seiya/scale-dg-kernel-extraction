#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dg_roof_bytes
#$ -j y
# H100 half of jobs/roof_bytes_ncu.sh -- the measurement that closes
# reports/h100_report.md 8.9.2's open item (H100 at p=767 sits at 46.8% of its
# DMMA roof and 18.1% of its bandwidth roof, and what binds it is unmeasured).
# The GB200 half is Slurm 83141, reports/measured_dram_and_roofs.md.
#
# Build on TSUBAME first (cc90 / sm_90):
#   module purge && module load nvhpc
#   make clean && make CUDA=1 GPUFLAGS=-gpu=cc90 GPUNVCCFLAGS=-arch=sm_90
#   qsub -g <TSUBAME_GROUP> jobs/tsubame_roof_bytes_ncu.sh
#
# SHAPE=16x8x4 because that is the instruction shape H100 needs at high order
# (reports/h100_report.md 8.3) and the shape its published winners use; 8x8x4
# would measure a configuration nobody runs on this machine.
#
# gpu_1=1 is one whole H100.  Do NOT use gpu_h / node_o (MIG instances).
set -u
export SHAPE="${SHAPE:-16x8x4}"
exec bash "${SGE_O_WORKDIR:-$PWD}/jobs/roof_bytes_ncu.sh"
