#!/bin/bash
#PJM -N dg_a100_emu
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# A100 CublasEmulation column -- measurement_inventory.md 19.7, FIRST item,
# and cublas_emulation_survey.md 9.3 ("H100 / A100 では依然として測っていない").
# Same shared body as jobs/tsubame_cublas_emu.sh; only the machine differences
# are here.
#
# BEFORE SUBMITTING (cc80 / sm_80):
#   module unload nvidia && module load nvidia/25.9   # do NOT load gcc/* or cuda/*
#   make clean
#   make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
#   fill in -g above, then:  pjsub jobs/wisteria_cublas_emu.sh
#
# THE TOOLCHAIN IS PART OF THE QUESTION HERE
# ------------------------------------------
# cuda_cublas_gemm.cu needs CUBLAS_VERSION >= 130002 (cuBLAS 13.0 update 2) for
# the emulation calls to exist at all; nvidia/25.9 is the newest module on
# Aquarius and may be below that line.  If it is, every emulated run returns
# NOT_SUPPORTED and the answer recorded is "the toolchain on this machine
# cannot express it", which the body separates from a hardware refusal by
# printing the compile-time CUBLAS_VER_* into metadata.txt.  Do not resubmit
# with a different compiler to make the cell fill in: nvidia/25.9 is the
# version every other A100 column was measured with (jobs/wisteria_paths.sh),
# and a mixed-compiler row would not belong in the same table.
#
# MEMORY: 40 GiB, and emulation adds a persistent 8 GiB workspace
# ---------------------------------------------------------------
# p=575's payload alone is 32.2 GiB (estimate_device_memory.py), so the
# emulated cell cannot fit and p=575/767 are out of ORDERS_HI here.  p=511 is
# the top order this column can carry, and even it may OOM emulated -- that is
# a result, printed with rc != 0, not a job failure.
#
# WALL CLOCK: share-short caps at 02:00:00 (pjsub SIM4720), which is why this
# is 3 rounds and validates only p=7 and p=255.  Emulation's cost on A100 is
# unknown -- it is what the job measures -- and on GB200 the p=7 emulated run
# is 300x the native one, so a generous sweep could exceed the cap on its own.
# If the p=511 precision point is wanted as well, submit a second, timing-free
# job once this one has shown what emulation costs here:
#   pjsub -x VALIDATE=1,VAL_ORDERS=511,TIMING=0 jobs/wisteria_cublas_emu.sh
set -u
export SCALE_DG_MACHINE=a100
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export EMU_MODES="nat emu"
export ORDERS_LO="7 255 511" ORDERS_HI=""
export ROUNDS=3 ROUNDS_HI=3
export VALIDATE="${VALIDATE:-1}" VAL_ORDERS="${VAL_ORDERS:-7 255}" TIMING="${TIMING:-1}"
export OUTDIR="${OUTDIR:-cublas_emu_a100_${PJM_JOBID:-manual}}"
exec bash "${PJM_O_WORKDIR:-$PWD}/jobs/cublas_emulation_ab.sh"
