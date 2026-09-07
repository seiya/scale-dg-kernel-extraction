#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=08:00:00
#$ -N dg_h100_emu
#$ -j y
#
# H100 CublasEmulation column -- measurement_inventory.md 19.7, FIRST item.
# GB200 has this from cublas_emulation_survey.md sections 8 and 9; H100 and
# A100 have never had it, and section 9.3 names them explicitly as unmeasured.
#
# All measurement logic is in the tracked shared body jobs/cublas_emulation_ab.sh,
# so the H100, A100 and GB200 columns are one measurement rather than three
# conventions.  This wrapper carries only what is machine-specific.
#
# BEFORE SUBMITTING (cc90 / sm_90):
#   module purge && module load nvhpc
#   make clean
#   make CUDA=1 GPUFLAGS=-gpu=cc90 GPUNVCCFLAGS=-arch=sm_90
#   qsub -g <TSUBAME_GROUP> jobs/tsubame_cublas_emu.sh
#
# gpu_1=1 is one complete H100.  Do NOT use gpu_h / node_o (MIG fractions).
#
# TWO WAYS THIS CAN COME BACK EMPTY, AND BOTH ARE RESULTS
# -------------------------------------------------------
# 1. The toolkit is older than cuBLAS 13.0u2, so cuda_cublas_gemm.cu compiled
#    the #else branch and every emulated run returns NOT_SUPPORTED.  That is a
#    toolchain finding, not a hardware one: the body prints the compile-time
#    CUBLAS_VER_* from the header into metadata.txt so the two cannot be
#    confused.  If it happens, check for a newer nvhpc module and resubmit.
# 2. cuBLAS accepts the calls and refuses sm_90 at run time.  That is the same
#    shape of finding as OZAKI2's sm_90/sm_80 illegal access
#    (ozaki2_implementation_report.md 8.7-8.13) -- a portability defect that
#    only a non-GB200 machine can show -- and it is worth the job either way.
#
# WALL CLOCK: 8 h against a GB200 cost of roughly 40 min for the same sweep
# (section 9 ran 6 rounds at p=7/255/511 and 3 at p=575/767 inside a 2 h cap,
# and p=7 emulated is 482 ms/stage there, which alone is ~29 s per run).  H100
# has no FP64-emulation number at all yet, so the margin is deliberate: the
# INT8 rate that emulation actually runs at is what this job is measuring.
# p=767 needs ~78 GiB of payload against 94 GB, plus the 8 GiB emulation
# workspace, so the emulated cell there may OOM.  That is a measurement; the
# body prints rc != 0 and continues.
# ADP: the GB200 column measured EmulationMantissaControl = "DYNAMIC" as a
# third mode (cublas_emulation_survey.md 10.4) and found it 2.88x FASTER than
# FIXED 55 at p=7 and bit-identical to the reference, while losing 1.14-1.80x
# at p >= 511.  H100 has not been asked.  To take it, override the mode list --
# SGE's -v cannot carry a space, so use colons:
#   qsub -g <GROUP> -v EMU_MODES=nat:emu:adp jobs/tsubame_cublas_emu.sh
# Every knob below is overridable the same way.
set -u
export SCALE_DG_MACHINE=h100
export EMU_MODES="${EMU_MODES:-nat emu}"
export ORDERS_LO="${ORDERS_LO:-7 255 511}" ORDERS_HI="${ORDERS_HI:-575 767}"
export ROUNDS="${ROUNDS:-6}" ROUNDS_HI="${ROUNDS_HI:-3}"
export VALIDATE="${VALIDATE:-1}" VAL_ORDERS="${VAL_ORDERS:-7 255 511}" TIMING="${TIMING:-1}"
export OUTDIR="${OUTDIR:-cublas_emu_h100_${JOB_ID:-manual}}"
exec bash "${SGE_O_WORKDIR:-$PWD}/jobs/cublas_emulation_ab.sh"
