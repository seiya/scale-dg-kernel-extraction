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
# WHAT IT MEASURES, AND THE TWO RUNTIMES
# ---------------------------------------------------------------------------
# p=63 only, every path, interleaved rounds, medians -- the same body and the
# same namelist substitution as every other column of the cross-machine table,
# so the numbers drop straight into it.  Validation runs first, because a
# compiler two years apart is exactly the kind of thing that changes an answer
# rather than a time.
#
# It sweeps each executable under TWO runtimes, because they answer different
# questions and the difference between them is itself the interesting number:
#
#   match   each build runs under the modules it was BUILT with (23.3 binary on
#           the 23.3 runtime).  This is what "using nvhpc 23.3" means in
#           practice -- compiler AND its cudart/cuBLAS -- and it is the honest
#           upper bound.
#   fixed   both builds run under nvidia/25.9.  Libraries are held constant, so
#           what is left is compiler CODE GENERATION alone.
#
#   match minus fixed  = the runtime libraries' share of the difference.
#
# Without the `match` arm this job silently measured only `fixed`: the body
# loads its modules once, so a 23.3-built binary was picking up 25.9's cudart
# and cuBLAS.  SCALE_DG_MODULE_UNLOAD is what makes switching actually take
# effect, since Aquarius has an nvidia module loaded by default and a plain
# `module load` of another version conflicts rather than replacing it.
#
# Reading it: whatever fraction moves under `match` is an upper bound on the
# compiler's contribution to the A100/GB200 ratios in a100_report.md section
# 3.1.  If it is under a percent or so, the compiler mismatch stops being a
# caveat.  If `fixed` is much smaller than `match`, the caveat is about
# libraries, not codegen, and should be described that way.
#
# Knobs: OLD_BUILD (default nv233), RUNTIMES (default "match fixed"),
#        ORDERS_A / VAL_ORDERS / ROUNDS as in the path sweep.
set -u
export SCALE_DG_MACHINE=a100
export SCALE_DG_MODULE_PURGE=0
export SCALE_DG_MODULE_UNLOAD="nvidia"
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
RUNTIMES="${RUNTIMES:-match fixed}"

# Which module each build was compiled with, so `match` can reproduce it.
modules_for() {
  case "$1" in
    nv259)  echo "nvidia/25.9" ;;
    nv2411) echo "nvidia/24.11" ;;
    nv2311) echo "nvidia/23.11" ;;
    nv233)  echo "nvidia/23.3" ;;
    *) echo "" ;;
  esac
}

ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1

missing=""
for build in nv259 "$OLD_BUILD"; do
  [ -x "$ROOT/scale-dg_extraction.$build" ] || missing="$missing $build"
  [ -n "$(modules_for "$build")" ] || { echo "ERROR: no module known for build $build; add it to modules_for()." >&2; exit 1; }
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
  make clean && make CUDA=1 GPUFLAGS='-gpu=cc80,cuda12.0' GPUNVCCFLAGS=-arch=sm_80
  cp scale-dg_extraction scale-dg_extraction.nv233

  module unload nvidia
  module load nvidia/25.9
  make clean && make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80

The last block leaves the tree in the state every other job expects.

nvidia/23.3 needs the cuda12.0 in GPUFLAGS: its nvcc is CUDA 12.0 but
nvfortran otherwise links a cudart without cudaLaunchKernelExC, which the PDL
launches in cuda_dg_kernels_tc.cu require, and the link fails with an
undefined reference to that symbol.

If a build still fails, widen the gap only as far as it will go -- try
nvidia/23.11, then 24.11, use the OLDEST that builds, and name the file after
it (OLD_BUILD=nv2311 pjsub ...).  A narrower gap still bounds the effect, just
less tightly; record which compiler was actually used.
RECIPE
  echo >&2 "missing:$missing"
  exit 1
fi

for runtime in $RUNTIMES; do
  for build in nv259 "$OLD_BUILD"; do
    case "$runtime" in
      match) mods="$(modules_for "$build")" ;;
      fixed) mods="nvidia/25.9" ;;
      *)     mods="$runtime" ;;
    esac
    echo
    echo "############ build=$build runtime=$runtime ($mods) ############"
    SCALE_DG_MODULES="$mods" \
    EXE="$ROOT/scale-dg_extraction.$build" \
    OUTDIR="a100_compiler_${build}_${runtime}_${PJM_JOBID:-manual}" \
      bash jobs/tsubame_paths.sh
  done
done
