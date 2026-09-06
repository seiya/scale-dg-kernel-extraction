#!/bin/bash
#PJM -N dg_a100_compiler_sens
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# How much of the A100 numbers is the compiler rather than the machine?
#
# reports/a100_report.md section 8.2 records the one variable this campaign
# could not control: the three machines ran three compilers -- nvhpc 26.3 on
# RIKYU, 26.1 on TSUBAME, 25.9 on Wisteria, because Aquarius has no 26.x.  That
# matters here more than it usually would: h100_report.md 8.7.3 traced the
# SPLIT reversal to nvfortran's register allocation and priced it at 13-20% on
# the kernel that carried it, so codegen is a first-order effect in this
# problem, not a rounding term.
#
# It cannot be removed, but it can be BOUNDED, on one machine, in one job: run
# the same order under two compilers that are far apart and see how much moves.
# nvidia/23.3 is Aquarius's default and about two and a half years older than
# 25.9, so it is the widest gap available and therefore an upper limit on what
# a 25.9-versus-26.3 difference could be worth.
#
# ---------------------------------------------------------------------------
# BEFORE SUBMITTING: build twice, freeze both
# ---------------------------------------------------------------------------
#   module unload nvidia
#   module load nvidia/25.9
#   make clean && make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
#   cp scale-dg_extraction scale-dg_extraction.nv259
#
#   module unload nvidia
#   module load nvidia/23.3
#   make clean && make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
#   cp scale-dg_extraction scale-dg_extraction.nv233
#
#   make clean && module unload nvidia && module load nvidia/25.9
#   make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80   # leave 25.9 built
#
#   pjsub jobs/wisteria_compiler_sens.sh
#
# If the 23.3 build fails, that is a result too -- record which file and why,
# and the bound stays open.  CUTLASS 4.7.0 wants a recent toolkit and 23.3
# carries CUDA 12.0/12.1; the paths under test here are the hand-written fused
# kernels, so a CUTLASS failure would still leave FUSED / FUSED_TC / FUSED_DFMA
# measurable.  In that case set PATHS accordingly and say so in the report.
#
# ---------------------------------------------------------------------------
# WHAT IT MEASURES
# ---------------------------------------------------------------------------
# p=63 only, both executables, every path, interleaved rounds, medians -- the
# same body and the same namelist substitution as every other column of the
# cross-machine table, so the numbers drop straight into it.  Validation runs
# first, because a two-and-a-half-year-old compiler is exactly the kind of
# thing that changes an answer rather than a time.
#
# Reading it: whatever fraction moves between 25.9 and 23.3 is an upper bound
# on the compiler's contribution to the A100/GB200 ratios in section 3.1.  If
# it is under a percent or so, the compiler mismatch stops being a caveat.
set -u
export SCALE_DG_MACHINE=a100
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export SCALE_DG_PEAK_FP64_CUDA=9.7
export SCALE_DG_PEAK_FP64_TENSOR=19.5
export SCALE_DG_PEAK_HBM=1.555
export SCALE_DG_RESOURCE="pjsub gpu=1 (A100 40GiB)"
export SCALE_DG_SHAPE_SPECS=
export ORDERS_A="${ORDERS_A:-63}"
export ORDERS_B=""
export ORDERS_C=""
export VAL_ORDERS="${VAL_ORDERS:-63}"
export ROUNDS="${ROUNDS:-6}"
# The older half of the comparison.  nvidia/23.3 is Aquarius's default and the
# widest gap available; fall back to a newer one only if 23.3 will not build.
OLD_BUILD="${OLD_BUILD:-nv233}"

ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1

missing=""
for build in nv259 "$OLD_BUILD"; do
  [ -x "$ROOT/scale-dg_extraction.$build" ] || missing="$missing $build"
done
if [ -n "$missing" ]; then
  cat >&2 <<'RECIPE'
ERROR: the frozen per-compiler executables are missing.

They are built on the LOGIN NODE, not here: this job only measures.  From the
checkout, one build per compiler, freezing each before the next make clean:

  module unload nvidia
  module load nvidia/25.9
  make clean && make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
  cp scale-dg_extraction scale-dg_extraction.nv259

  module unload nvidia
  module load nvidia/23.3
  make clean && make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
  cp scale-dg_extraction scale-dg_extraction.nv233

  module unload nvidia
  module load nvidia/25.9
  make clean && make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80

The last block leaves the tree in the state every other job expects.

If the 23.3 build fails, it will fail in cuda_cutlass_gemm_fused.cu: CUTLASS
4.7.0 wants a recent toolkit and 23.3 carries CUDA 12.0/12.1.  Nothing links
without that object, so the answer is not to drop paths but to widen the gap
as far as it will go -- try nvidia/23.11, then 24.11, and use the OLDEST that
builds, naming the file after it (OLD_BUILD=nv2311 pjsub ...).  A narrower gap
still gives an upper bound, just a weaker one; record which compiler was
actually used.
RECIPE
  echo >&2 "missing:$missing"
  exit 1
fi

for build in nv259 "$OLD_BUILD"; do
  echo
  echo "############ compiler build: $build ############"
  EXE="$ROOT/scale-dg_extraction.$build" \
  OUTDIR="a100_compiler_${build}_${PJM_JOBID:-manual}" \
    bash jobs/tsubame_paths.sh
done
