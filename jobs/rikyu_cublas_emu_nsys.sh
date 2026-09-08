#!/bin/bash
#SBATCH -J dg_gb200_emunsys
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH -o slurm-%j.out
#
# GB200 half of the question cublas_emulation_survey.md 10.2 left open:
# emulation WINS at p >= 511 on GB200 (emu/nat 0.947 at p=511, 0.915 at p=767)
# and LOSES on H100 (2.18 and 2.47), while the NATIVE side of the same
# comparison is only 10.7% apart.  The difference is inside the emulated path,
# and no job has looked at what that path actually runs.
#
# This traces it by KERNEL NAME on each machine, through the same tracked body
# as the native library question (jobs/cublas_kernel_names_nsys.sh, KINDS knob
# added 2026-09-08).  Run the H100 half with jobs/tsubame_cublas_emu_nsys.sh;
# the two columns are the same script, which is what makes them comparable.
#
# Why kernel names and not another timing: the timing is already measured
# (survey section 10) and it does not say WHICH kernels differ.  A cuBLAS
# FP64 emulation call is a sequence -- split into INT8 slices, INT8 GEMMs,
# reconstruction -- and the machines can differ in any of the three.
#
# BEFORE SUBMITTING:
#   module load nvhpc-hpcx
#   make CUDA=1 GPUFLAGS=-gpu=cc100
#   sbatch jobs/rikyu_cublas_emu_nsys.sh
#
# nsys, not ncu, and via sbatch: AGENTS.md forbids running either profiler on
# the login node, DEBUGINFOD_URLS= and --resolve-symbols=false are set by the
# body, and the namelists it writes force UseCudaGraph = .false. because nsys
# hangs on the graph path.
set -u
ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# The KINDS knob landed in the shared body on 2026-09-08.  A checkout that
# predates it IGNORES this wrapper's KINDS and silently traces GEMM/GEMM_CUTE
# instead of the emulated path -- SGE 8602137 did exactly that and produced a
# clean-looking report answering the wrong question.  Fail loudly instead.
BODY="$ROOT_DIR/jobs/cublas_kernel_names_nsys.sh"
if ! grep -q 'readonly KINDS=' "$BODY"; then
  echo "ERROR: $BODY predates the KINDS knob (2026-09-08)." >&2
  echo "  Copy the current jobs/cublas_kernel_names_nsys.sh to this machine" >&2
  echo "  and resubmit; without it this job traces the wrong variants." >&2
  exit 1
fi
export SCALE_DG_MODULES="nvhpc-hpcx"
export KINDS="GEMM GEMM_EMU"
export ORDERS="${ORDERS:-255 511}"
export NSTEP="${NSTEP:-10}"
export OUTDIR="${OUTDIR:-cublas_emu_nsys_gb200_${SLURM_JOB_ID:-manual}}"
exec bash "$BODY"
