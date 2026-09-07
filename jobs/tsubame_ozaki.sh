#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=04:00:00
#$ -N dg_h100_ozaki
#$ -j y
#
# H100 Ozaki column -- measurement_inventory.md 19.7, third item.
# GB200 has this from job 83482 (section 18.2, part B); H100 and A100 never
# had it.  This drives the TRACKED body jobs/tsubame_paths.sh through its
# SCALE_DG_PATHS_<order> knob, so no measurement logic lives here: the
# columns stay one measurement across the three machines.
#
# BEFORE SUBMITTING (cc90 / sm_90):
#   module purge && module load nvhpc
#   make clean
#   make CUDA=1 GPUFLAGS=-gpu=cc90 GPUNVCCFLAGS=-arch=sm_90
#   qsub -g <TSUBAME_GROUP> jobs/tsubame_ozaki.sh
#
# Ozaki needs slice/modulus workspaces on top of the fields.  If a high order
# OOMs that is a RESULT, not a job failure: the body prints rc != 0 and goes
# on.  p=1023 is excluded because GB200 could not hold it either (18.10).
set -u
P="CUDAFORTRAN_GEMM:CUDAFORTRAN_GEMM_OZAKI1:CUDAFORTRAN_GEMM_OZAKI2"
for p in 7 15 31 63 127 255 511 575 767; do export SCALE_DG_PATHS_$p="$P"; done
export ORDERS_A="7 15 31 63 127 255" ORDERS_B="511" ORDERS_C="575 767"
export ROUNDS=6 ROUNDS_HI=6 ROUNDS_XHI=3
export VALIDATE=0 TIMING=1
export OUTDIR="${OUTDIR:-tsubame_ozaki_${JOB_ID:-manual}}"
exec bash "${SGE_O_WORKDIR:-$PWD}/jobs/tsubame_paths.sh"
