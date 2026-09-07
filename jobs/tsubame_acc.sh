#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dg_h100_acc
#$ -j y
#
# H100 OpenACC column -- measurement_inventory.md 19.7, third item.
# GB200 has this from job 83485 (section 18.2, part C).
#
# THIS NEEDS A DIFFERENT BUILD from every other job here.  OpenACC and CUDA
# Fortran use incompatible objects and module files, so build it in a
# WORKTREE and leave the CUDA build in place:
#
#   git worktree add --detach ../dg-acc HEAD
#   cd ../dg-acc && ln -s <main>/third_party third_party
#   module purge && module load nvhpc
#   make ACC=1 GPUFLAGS=-gpu=cc90
#   qsub -g <TSUBAME_GROUP> -v EXE=$PWD/scale-dg_extraction \
#        <main>/jobs/tsubame_acc.sh
#
# VALIDATE=0 is not laziness: an ACC build has no CUDAFORTRAN_* paths, so the
# body's reference kernel does not exist in it.  The numerical check for these
# two paths is the CUDA jobs' business (section 18.2).
set -u
# EXE must be an ABSOLUTE path to the OpenACC executable.  Do not pass $PWD
# through the scheduler: job 9539009 was submitted that way, the variable
# arrived empty, EXE became "/scale-dg_extraction", and all 24 runs failed
# with "No such file or directory".  Write the path out, and this checks it.
if [ -z "${EXE:-}" ]; then
  echo "ERROR: set EXE to the OpenACC executable (absolute path)." >&2
  echo "   e.g.  qsub -v EXE=/work/.../dg-acc/scale-dg_extraction ..." >&2
  exit 2
fi
case "$EXE" in /*) ;; *) echo "ERROR: EXE must be absolute, got '$EXE'" >&2; exit 2;; esac
[ -x "$EXE" ] || { echo "ERROR: EXE is not an executable file: '$EXE'" >&2; exit 2; }
echo "EXE=$EXE"
export EXE
for p in 7 15; do export SCALE_DG_PATHS_$p="OPENACC_ASIS:OPENACC_SPLIT"; done
export ORDERS_A="7 15" ORDERS_B="" ORDERS_C=""
export ROUNDS=6 VALIDATE=0 TIMING=1
export OUTDIR="${OUTDIR:-tsubame_acc_${JOB_ID:-manual}}"
exec bash "${SGE_O_WORKDIR:-$PWD}/jobs/tsubame_paths.sh"
