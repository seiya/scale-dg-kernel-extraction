#!/bin/bash
#PJM -N dg_a100_paths
#PJM -L rscgrp=CHANGE_ME
#PJM -L gpu=1
#PJM -L elapse=8:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# The A100 column of the cross-machine table: jobs/tsubame_paths.sh, unchanged,
# under Fujitsu TCS instead of SGE/Slurm.  Running the SAME body on all three
# machines is what makes the ratios one measurement rather than three
# conventions -- same namelists, same interleaving, same timer, same median.
#
# Machine: Wisteria/BDEC-01 Aquarius (U. Tokyo), NVIDIA A100 SXM 40 GiB,
#          Xeon Platinum 8360Y x2 (Ice Lake), 8 GPUs per node.
# Purpose: answer reports/a100_prediction.md, which was committed BEFORE this
#          job ran.  Do not edit that report's prediction tables; put the
#          results beside them.
#
# ---------------------------------------------------------------------------
# BEFORE SUBMITTING
# ---------------------------------------------------------------------------
# 1. Fill in the two CHANGE_ME above:
#      -L rscgrp=   the Aquarius resource group.  List what you may use with
#                   `pjshowrsc` or the centre's user guide; a debug/short group
#                   is fine for a smoke test, but the full sweep needs ~8 h.
#      -g           your project/group ID.
#
# 2. Build for A100.  cc80 is NOT the tree's default target and needs both
#    guards that landed in 9853e28 (PDL and CutlassMmaShape); an older checkout
#    will not compile cuda_dg_kernels_tc.cu for sm_80 at all.
#
#      module load nvidia/25.9      # closest available to RIKYU 26.3 / TSUBAME 26.1
#      module load gcc/12.2.0       # NOT the default gcc/8.3.1 -- CUTLASS names
#                                   # GCC 8.5 for known regressions (README.md:151)
#      # do NOT load cuda/* : nvhpc brings its own and mixing breaks the build
#      make clean
#      make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
#
#    Why 25.9 and not the default 23.3: the compiler is not a neutral variable
#    here.  h100_report.md 8.7.3 traced the SPLIT reversal to nvfortran's
#    register allocation and priced it at 13-20% on the kernel that carried it.
#    Dropping to 23.3 stacks ~2.5 years of codegen drift on top of the
#    architecture difference this campaign exists to isolate.
#
# 3. Submit FROM the checkout (the job writes its output tree into the working
#    copy and finds it through PJM_O_WORKDIR):
#
#      pjsub jobs/wisteria_paths.sh
#
#    From anywhere else, name the checkout explicitly:
#      pjsub -x SCALE_DG_ROOT=/path/to/checkout /path/to/checkout/jobs/wisteria_paths.sh
#
# ---------------------------------------------------------------------------
# WHAT THIS JOB DOES NOT SWEEP, AND WHY
# ---------------------------------------------------------------------------
# * CutlassMmaShape 16x8x4 / 16x8x8 / 16x8x16.  These are sm_90 instructions.
#   CUTLASS compiles them for sm_80 with no warning but turns them into
#   brkpt / assert(0), so the tree now refuses them with error stop
#   (a100_prediction.md 7.2).  SCALE_DG_SHAPE_SPECS= drops them from the sweep;
#   A100 runs the default 8x8x4 only, which is also all GB200 has.  That makes
#   the A100/GB200 comparison one instruction, not two.
# * p=767 and p=1023.  76.2 GiB and 180 GiB of payload
#   (estimate_device_memory.py) against 40 GiB.  p=575 needs 32.2 GiB, which
#   fits at 81% of the card -- expected to run, but if it OOMs that is a
#   result, not a job failure: rc != 0 is printed and the sweep continues.
# * FUSED_TC / FUSED_DFMA carry a known handicap here: Programmatic Dependent
#   Launch is sm_90-only, so on A100 the hints are compiled out and the
#   launches fall back to ordinary stream order.  That tax was measured on
#   GB200 first, exactly so this run stays interpretable:
#   p=63 FUSED_TC 2.98%, p=31 FUSED_TC 1.14%, p=63 FUSED_DFMA 1.69%,
#   p=31 FUSED_DFMA 0 (a100_prediction.md 7.1.1).  p=127's PDL has no stage
#   macro and could not be switched off, so its tax is unknown.
#
# ---------------------------------------------------------------------------
# KNOBS (all optional; export before pjsub, or use -x)
# ---------------------------------------------------------------------------
#   ROUNDS=6 ROUNDS_HI=6 ROUNDS_XHI=3   interleaved rounds per part
#   ORDERS_A/B/C=...                    orders per part ("" skips the part)
#   VALIDATE=1 TIMING=1                 phases
#   OUTDIR=...                          output tree
# For a smoke test in a debug group, try:
#   ROUNDS=1 ORDERS_A="7 63" ORDERS_B= ORDERS_C= pjsub jobs/wisteria_paths.sh
set -u

export SCALE_DG_MACHINE=a100

# Wisteria module names.  The shared body defaults to `nvhpc`, which does not
# exist here.  module purge is skipped: the site's default environment carries
# more than the compiler.
export SCALE_DG_MODULES="nvidia/25.9 gcc/12.2.0"
export SCALE_DG_MODULE_PURGE=0

# Roofs printed into metadata.txt so no report has to guess them later.
# A100 SXM 40 GiB: 9.7 TF FP64 CUDA core, 19.5 TF FP64 Tensor Core (2x, unlike
# GB200 where the two are equal), 1.555 TB/s HBM2e.
export SCALE_DG_PEAK_FP64_CUDA=9.7
export SCALE_DG_PEAK_FP64_TENSOR=19.5
export SCALE_DG_PEAK_HBM=1.555
export SCALE_DG_RESOURCE="pjsub gpu=1 (A100 40GiB)"

# No sm_90 MMA shapes on Ampere.  Set-but-empty, which the body honours.
export SCALE_DG_SHAPE_SPECS=

# p=767 does not fit in 40 GiB; p=575 does, at 81%.
export ORDERS_C="${ORDERS_C-575}"

export OUTDIR="${OUTDIR:-a100_paths_${PJM_JOBID:-manual}}"

ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/tsubame_paths.sh
