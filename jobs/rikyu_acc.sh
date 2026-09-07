#!/bin/bash
#SBATCH --job-name=dg_gb200_acc
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH -o slurm-%j.out
#
# The GB200 OpenACC column on the SAME axis as the H100 and A100 ones.
# measurement_inventory.md 18.8 measured these paths on the
# "Volume derivate + surface lift + Element boundary flux" axis, and the
# H100/A100 jobs (section 21.4) report "Step loop per stage", so the three
# machines could not be divided.  This takes GB200 through the same tracked
# body, jobs/tsubame_paths.sh, that the other two used.
#
# Needs the OpenACC build, which is incompatible with the CUDA one and so
# lives in a worktree:
#   git worktree add --detach .claude/worktrees/acc HEAD
#   cd .claude/worktrees/acc && ln -s <main>/third_party third_party
#   module load nvhpc && make ACC=1 GPUFLAGS=-gpu=cc100
#   sbatch --export=ALL,EXE=<abs path>/scale-dg_extraction jobs/rikyu_acc.sh
set -u
if [ -z "${EXE:-}" ]; then
  echo "ERROR: set EXE to the OpenACC executable (absolute path)." >&2
  echo "   e.g.  sbatch --export=ALL,EXE=/path/to/acc/scale-dg_extraction jobs/rikyu_acc.sh" >&2
  exit 2
fi
case "$EXE" in /*) ;; *) echo "ERROR: EXE must be absolute, got '$EXE'" >&2; exit 2;; esac
[ -x "$EXE" ] || { echo "ERROR: EXE is not an executable file: '$EXE'" >&2; exit 2; }
echo "EXE=$EXE"
export EXE
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
export SCALE_DG_MACHINE=gb200
export SCALE_DG_MODULES="nvhpc"
for p in 7 15; do export SCALE_DG_PATHS_$p="OPENACC_ASIS:OPENACC_SPLIT"; done
export ORDERS_A="7 15" ORDERS_B="" ORDERS_C=""
export ROUNDS=6 VALIDATE=0 TIMING=1
export OUTDIR="${OUTDIR:-gb200_acc_${SLURM_JOB_ID:-manual}}"
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/tsubame_paths.sh
