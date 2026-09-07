#!/bin/bash
#SBATCH -J dg_gb200_emu
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out
#
# GB200 CublasEmulation column, re-taken through the tracked shared body.
#
# WHY RE-TAKE A COLUMN THAT ALREADY EXISTS
# ----------------------------------------
# cublas_emulation_survey.md section 9 measured this on the current tree
# (Slurm 84461), but from an untracked job, and its in-job numerical table was
# WRONG -- the comparison read the text dumps as packed doubles and returned
# 1.092e+25 at three orders, so section 9.2's table had to be redone by hand on
# the login node (section 9.0, boxed note).  The H100 and A100 columns are new,
# and a machine-to-machine ratio is only reproducible if every column's script
# is tracked (AGENTS.md, "jobs/").  This puts GB200 on the same body.
#
# It also closes the second item of section 9.3: EmulationMantissaControl =
# "DYNAMIC" (ADP) has not been measured since sections 4 and 5, several trees
# ago.  EMU_MODES carries it here as a third interleaved mode.  Section 6 found
# ADP selecting 54 bits at p=1023 while FIXED 55 lost 1.41x on the y GEMM, so
# the two are not interchangeable and the current tree should say so with a
# number.
#
# And it attempts the first item of 9.3: p=1023 emulated.  Section 8 could not
# hold the persistent 8 GiB workspace beside a 176 GiB field payload
# (p1023_gap_study.md), and the current tree has never been asked.  An OOM here
# is the recorded answer, not a job failure; the body prints rc != 0 and goes on.
#
# BEFORE SUBMITTING:
#   module load nvhpc-hpcx
#   make clean && make CUDA=1 GPUFLAGS=-gpu=cc100
#   sbatch jobs/rikyu_cublas_emu.sh
#
# --exclusive because section 9's table is a ratio between two runs whose
# difference is 300x at p=7 and 5% at p=511: the small end needs a quiet node.
set -u
export SCALE_DG_MACHINE=gb200
export SCALE_DG_MODULES="nvhpc-hpcx"
export EMU_MODES="nat emu adp"
export ORDERS_LO="7 255 511" ORDERS_HI="575 767 1023"
export ROUNDS=6 ROUNDS_HI=3
export VALIDATE=1 VAL_ORDERS="7 255 511" TIMING=1
export OUTDIR="${OUTDIR:-cublas_emu_gb200_${SLURM_JOB_ID:-manual}}"
exec bash "${SLURM_SUBMIT_DIR:-$PWD}/jobs/cublas_emulation_ab.sh"
