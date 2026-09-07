#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -N dg_h100_emunsys
#$ -j y
#
# H100 half of cublas_emulation_survey.md 10.2's open question: which kernels
# does the emulated volume GEMM actually run, and how do they differ from the
# GB200 ones?  Emulation loses 2.18x at p=511 here and wins (0.947) there,
# while native is only 10.7% apart, so the difference is inside the emulated
# path.  Same tracked body and same KINDS as jobs/rikyu_cublas_emu_nsys.sh.
#
# BEFORE SUBMITTING (cc90 / sm_90):
#   module purge && module load nvhpc
#   make CUDA=1 GPUFLAGS=-gpu=cc90 GPUNVCCFLAGS=-arch=sm_90
#   qsub -g <TSUBAME_GROUP> jobs/tsubame_cublas_emu_nsys.sh
#
# gpu_1=1 is one complete H100; do NOT use the MIG fractions.
# A100 is not in this comparison at all: its cuBLAS is 12.9.1 and cannot
# express emulation (survey 10.3).
set -u
export KINDS="GEMM GEMM_EMU"
export ORDERS="${ORDERS:-255 511}"
export NSTEP="${NSTEP:-10}"
export OUTDIR="${OUTDIR:-cublas_emu_nsys_h100_${JOB_ID:-manual}}"
exec bash "${SGE_O_WORKDIR:-$PWD}/jobs/cublas_kernel_names_nsys.sh"
