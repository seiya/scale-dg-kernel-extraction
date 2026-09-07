#!/bin/bash
#
# Shared body: cuBLAS FP64 emulation (native vs FIXED / DYNAMIC) A/B.
#
# WHY THIS IS A SHARED BODY AND NOT THREE SCRIPTS
# -----------------------------------------------
# measurement_inventory.md 19.7, first item: `CublasEmulation` is measured on
# GB200 only (cublas_emulation_survey.md sections 8 and 9); H100 and A100 have
# never had it.  Closing that row produces a machine-to-machine ratio, and
# AGENTS.md ("jobs/" in Important Files) requires every column of such a ratio
# to come from one tracked script.  This file is that script; the per-scheduler
# wrappers (jobs/{rikyu,tsubame,wisteria}_cublas_emu.sh) carry only the machine
# differences -- scheduler directives, module names, orders that fit in memory.
#
# It also re-takes the GB200 column, on purpose.  Section 9 was measured by an
# untracked job whose in-job validation READ THE DUMPS AS BINARY when they are
# text (section 9.0, boxed note), so the numerical table there had to be redone
# by hand on the login node.  The comparison here is the text-reading `cmp_dump`
# of jobs/tsubame_paths.sh, so the three columns share that too.
#
# WHAT IS MEASURED
# ----------------
# One namelist per order -- namelists/perf_p<order>_gemm.conf -- with ONLY the
# emulation knobs substituted.  Ne, PolyOrder, dt, nstep, UseCudaGraph and
# DqdtKernel_Type are untouched, so the native and emulated runs differ in one
# thing (AGENTS.md, Profiling).  Modes:
#
#   nat   CublasEmulation = .false.                       (native FP64 cuBLAS)
#   emu   CublasEmulation = .true.,  FIXED 55 bits        (the tree default)
#   adp   CublasEmulation = .true.,  DYNAMIC (ADP) 55 max
#
# `adp` is off by default: cublas_emulation_survey.md section 9.3 records it as
# unmeasured on the current tree, so the wrapper that wants it says
# EMU_MODES="nat emu adp".  Rounds are interleaved so a drift hits every mode.
#
# TOOLCHAIN, NOT ONLY HARDWARE
# ----------------------------
# cuda_cublas_gemm.cu guards the emulation calls with CUBLAS_VERSION >= 130002
# (cuBLAS 13.0 update 2).  An older toolkit compiles fine and then returns
# CUBLAS_STATUS_NOT_SUPPORTED at cublas_gemm_init, and a newer cuBLAS may still
# refuse an older architecture at run time.  Those are different findings, so
# the job records the compile-time cuBLAS version from the header AND the
# executable's own startup line, and a failure is printed and stepped over
# rather than ending the sweep.  "It does not run on sm_80" is a result of the
# same kind as OZAKI2's sm_90/sm_80 defect (ozaki2_implementation_report.md
# 8.7-8.13) -- and, like that one, only visible off GB200.
#
# MEMORY
# ------
# Emulation holds ONE persistent 8 GiB workspace (cuda_cublas_gemm.cu,
# kEmulationWorkspaceBytes) on top of the fields, so an order that fits
# natively can still OOM emulated.  Expected on A100 at p=575 (32.2 GiB of
# payload against 40 GiB) and possible on H100 at p=767 (~78 GiB against 94).
# rc != 0 is printed and the sweep continues; that cell is a measurement.
#
# KNOBS (export before submitting, or use the scheduler's -v / -x)
# ----------------------------------------------------------------
#   EMU_MODES="nat emu"        modes, interleaved in this order
#   ORDERS_LO="7 255 511"      orders swept ROUNDS times
#   ORDERS_HI="575 767"        orders swept ROUNDS_HI times ("" skips)
#   ROUNDS=6 ROUNDS_HI=3
#   VALIDATE=1 VAL_ORDERS="7 255 511"
#   TIMING=1
#   OUTDIR=... EXE=...
#   SCALE_DG_MODULES / SCALE_DG_MODULE_PURGE / SCALE_DG_MODULE_UNLOAD
#   SCALE_DG_MACHINE=gb200|h100|a100   (names the frozen executable copy)
# Separators ":" and "," are accepted wherever a list is, because SGE's -v can
# carry neither a space nor a comma inside one value.
set -u

#-----------------------------------------------------------------------------
# Locate the checkout.  NOT from $0: SGE runs a copy out of the execution
# host's spool (jobs/tsubame_paths.sh explains this at length).
#-----------------------------------------------------------------------------
ROOT=""
for candidate in "${SCALE_DG_ROOT:-}" "${SGE_O_WORKDIR:-}" "${PJM_O_WORKDIR:-}" \
                 "${SLURM_SUBMIT_DIR:-}" "$PWD" \
                 "$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; do
  [ -n "$candidate" ] || continue
  [ -f "$candidate/namelists/perf_p7_gemm.conf" ] || continue
  [ -f "$candidate/Makefile" ] || continue
  [ -w "$candidate" ] || continue
  ROOT="$candidate"
  break
done
if [ -z "$ROOT" ]; then
  echo "ERROR: cannot find a writable scale-dg-kernel-extraction checkout." >&2
  echo "  submit from the checkout, or pass SCALE_DG_ROOT=/path/to/checkout." >&2
  exit 1
fi
cd "$ROOT" || exit 1
echo "root=$ROOT"

readonly MODULES=${SCALE_DG_MODULES:-nvhpc}
readonly MODULE_PURGE=${SCALE_DG_MODULE_PURGE:-1}
readonly MODULE_UNLOAD=${SCALE_DG_MODULE_UNLOAD:-}
[ "$MODULE_PURGE" = 1 ] && module purge
for m in $MODULE_UNLOAD; do module unload "$m" 2>/dev/null || true; done
for m in $MODULES; do module load "$m"; done

list() { printf '%s' "$1" | tr ':,' '  '; }
readonly EMU_MODES=$(list "${EMU_MODES-nat emu}")
readonly ORDERS_LO=$(list "${ORDERS_LO-7 255 511}")
readonly ORDERS_HI=$(list "${ORDERS_HI-575 767}")
readonly VAL_ORDERS=$(list "${VAL_ORDERS-7 255 511}")
readonly ROUNDS=${ROUNDS:-6}
readonly ROUNDS_HI=${ROUNDS_HI:-3}
readonly VALIDATE=${VALIDATE:-1}
readonly TIMING=${TIMING:-1}
readonly JOB_TAG=${JOB_ID:-${SLURM_JOB_ID:-${PJM_JOBID:-manual}}}
readonly MACHINE=${SCALE_DG_MACHINE:-h100}
readonly OUTDIR=${OUTDIR:-"cublas_emu_${MACHINE}_${JOB_TAG}"}
readonly N="$ROOT/namelists"

mkdir -p "$OUTDIR/config" "$OUTDIR/run_logs" "$OUTDIR/scratch" || exit 1

# Measure a frozen copy: a concurrent make must not relink under the job.
if [ -n "${EXE:-}" ]; then
  EXE="$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")"
else
  if [ ! -x ./scale-dg_extraction ]; then
    echo "ERROR: ./scale-dg_extraction is missing; build first (see the wrapper)." >&2
    exit 1
  fi
  cp -p ./scale-dg_extraction "$OUTDIR/scale-dg_extraction.$MACHINE"
  EXE="$(cd "$OUTDIR" && pwd)/scale-dg_extraction.$MACHINE"
fi
readonly EXE

# The compile-time cuBLAS version decides whether the emulation calls exist at
# all, so record it rather than inferring it from a failure later.
cublas_header_version() {
  local nvcc_bin hdr
  nvcc_bin=$(command -v nvcc 2>/dev/null) || true
  for hdr in \
      "${nvcc_bin:+$(dirname "$nvcc_bin")/../include/cublas_api.h}" \
      "${CUDA_HOME:-}/include/cublas_api.h" \
      "${NVHPC_ROOT:-}/math_libs/include/cublas_api.h"; do
    [ -n "$hdr" ] && [ -f "$hdr" ] || continue
    awk '/#define CUBLAS_VER_(MAJOR|MINOR|PATCH|BUILD)/{printf "%s=%s ", $2, $3}
         END{printf "\n"}' "$hdr"
    echo "  header=$hdr"
    return 0
  done
  echo "unknown (no cublas_api.h found on this toolchain)"
}

{
  echo "date=$(date --iso-8601=seconds)"
  echo "job_id=$JOB_TAG"
  echo "host=$(hostname)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "exe=$EXE"
  echo "machine=$MACHINE"
  echo "emu_modes=$EMU_MODES"
  echo "orders_lo=$ORDERS_LO orders_hi=$ORDERS_HI"
  echo "rounds=$ROUNDS rounds_hi=$ROUNDS_HI validate=$VALIDATE val_orders=$VAL_ORDERS"
  echo
  echo "cublas_compile_version:"; cublas_header_version
  echo
  echo "gpu:"
  nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.max.sm \
    --format=csv,noheader 2>&1
  echo
  echo "nvfortran_version:"; nvfortran --version 2>&1 | head -n 4
  echo
  echo "nvcc_version:"; nvcc --version 2>&1 | tail -n 2
  echo
  echo "module_list:"; module list 2>&1
  echo
  echo "git_status:"; git status --short 2>/dev/null | head -n 40
} > "$OUTDIR/metadata.txt"
cat "$OUTDIR/metadata.txt"

#-----------------------------------------------------------------------------
# Config generation.  One template per order; only the emulation knobs move.
# The knobs are appended when the template lacks them, because main.f90 gives
# them defaults (FIXED / 55) and older namelists do not spell them out.
#-----------------------------------------------------------------------------
mkconf() { # $1 out  $2 order  $3 mode  [$4 nstep override]  [$5 kernel override]
  local out=$1 order=$2 mode=$3 nstep=${4:-} kernel=${5:-}
  local template="$N/perf_p${order}_gemm.conf" emu ctrl
  if [ ! -f "$template" ]; then
    echo "ERROR: no template $template" >&2
    return 1
  fi
  case "$mode" in
    nat) emu=".false." ctrl="FIXED" ;;
    emu) emu=".true."  ctrl="FIXED" ;;
    adp) emu=".true."  ctrl="DYNAMIC" ;;
    *)   echo "ERROR: unknown mode $mode" >&2; return 1 ;;
  esac
  sed -e "s/CublasEmulation *= *\.\(true\|false\)\./CublasEmulation = $emu/" \
      "$template" > "$out" || return 1
  grep -q "CublasEmulation" "$out" || \
    sed -i "s#^/#  CublasEmulation = $emu\n/#" "$out"
  if grep -q "EmulationMantissaControl" "$out"; then
    sed -i "s/EmulationMantissaControl *= *\"[A-Z]*\"/EmulationMantissaControl = \"$ctrl\"/" "$out"
  else
    sed -i "s#^/#  EmulationMantissaControl = \"$ctrl\"\n  EmulationMantissaBits = 55\n/#" "$out"
  fi
  [ -n "$nstep" ] && sed -i -e "s/nstep *= *[0-9]*/nstep = $nstep/" \
                            -e "s/output_interval *= *[0-9]*/output_interval = $nstep/" "$out"
  [ -n "$kernel" ] && sed -i "s/DqdtKernel_Type *= *\"[A-Z0-9_]*\"/DqdtKernel_Type = \"$kernel\"/" "$out"
  return 0
}

# The dumps are TEXT, one value per line (SCALE_DG_DUMP_DQDT).  Reading them as
# packed doubles is what made the in-job table of section 9 wrong; this is the
# reader jobs/tsubame_paths.sh uses, extended to count Inf/NaN because 941b066
# fixed exactly that failure on this path.
cmp_dump() { # $1 ref  $2 run
  python3 - "$1" "$2" <<'PY'
import sys, math
worst = ref = 0.0
n = bad = 0
with open(sys.argv[1]) as a, open(sys.argv[2]) as b:
    for la, lb in zip(a, b):
        x = float(la); y = float(lb)
        n += 1
        if math.isinf(y) or math.isnan(y):
            bad += 1
            continue
        d = abs(x - y)
        if d > worst: worst = d
        if abs(x) > ref: ref = abs(x)
print("n=%d max|diff|=%.3e ref_max=%.3e rel=%.3e inf_or_nan=%d"
      % (n, worst, ref, worst / ref if ref else 0.0, bad))
PY
}

#-----------------------------------------------------------------------------
# Validation.  A new machine, a new toolchain and a different cuBLAS emulation
# kernel are not covered by the GB200 record.  Point-varying coefficients, full
# owned dqdt, native reference at the same order (AGENTS.md, Validation).
# Section 9.2 found FIXED 55 is NOT roundoff-equivalent and degrades with
# K = Nq, so the answer here is part of the result, not a gate.
#-----------------------------------------------------------------------------
if [ "$VALIDATE" = 1 ]; then
  echo
  echo "===== VALIDATION ====="
  for p in $VAL_ORDERS; do
    if [ "$p" -le 127 ]; then ref_kernel=CUDAFORTRAN_SPLIT; else ref_kernel=CUDAFORTRAN_GEMM; fi
    ref_conf="$OUTDIR/config/val_p${p}_ref.conf"
    mkconf "$ref_conf" "$p" nat 1 "$ref_kernel" || continue
    ref_dump="$OUTDIR/scratch/dq_p${p}_ref"
    if ! env SCALE_DG_VARYING_COEFF=1 SCALE_DG_DUMP_DQDT="$ref_dump" \
        timeout 1800 "$EXE" "$ref_conf" > "$OUTDIR/run_logs/val_p${p}_ref.log" 2>&1; then
      echo "p=$p  REFERENCE FAILED ($ref_kernel); skipping"
      sed 's/^/    | /' "$OUTDIR/run_logs/val_p${p}_ref.log" | tail -n 6
      rm -f "$ref_dump"
      continue
    fi
    for mode in $EMU_MODES; do
      [ "$mode" = nat ] && continue
      conf="$OUTDIR/config/val_p${p}_${mode}.conf"
      mkconf "$conf" "$p" "$mode" 1 || continue
      dump="$OUTDIR/scratch/dq_p${p}_${mode}"
      if env SCALE_DG_VARYING_COEFF=1 SCALE_DG_DUMP_DQDT="$dump" \
          timeout 1800 "$EXE" "$conf" > "$OUTDIR/run_logs/val_p${p}_${mode}.log" 2>&1; then
        printf "p=%-4s %-4s vs %-18s %s\n" "$p" "$mode" "$ref_kernel" \
          "$(cmp_dump "$ref_dump" "$dump")"
        grep -h "cuBLAS FP64 emulation" "$OUTDIR/run_logs/val_p${p}_${mode}.log" \
          | sed 's/^/    startup: /'
      else
        printf "p=%-4s %-4s vs %-18s RUN FAILED\n" "$p" "$mode" "$ref_kernel"
        tail -n 6 "$OUTDIR/run_logs/val_p${p}_${mode}.log" | sed 's/^/    | /'
      fi
      rm -f "$dump"
    done
    rm -f "$ref_dump"
  done | tee "$OUTDIR/validation.txt"
  echo "VALIDATION_DONE"
fi

#-----------------------------------------------------------------------------
# Timing.  Device-event tendency time is the axis of section 8/9's table
# (ms/stage = "CUDA device GEMM tendency" / (measured steps x 3)); the wall
# columns are kept beside it as AGENTS.md asks.
#-----------------------------------------------------------------------------
run_one() { # $1 tag  $2 conf  $3 run-log
  local out rc dev
  out=$(timeout 3600 "$EXE" "$2" 2>&1); rc=$?
  printf '%s\n' "$out" > "$3"
  dev=$(printf '%s\n' "$out" | awk '/CUDA device/{s+=$NF} END{if (s>0) printf "%.5E", s; else printf "-"}')
  printf "%-34s rc=%-3s steps=%-6s devsum=%-12s stage=%-12s main=%s\n" "$1" "$rc" \
    "$(printf '%s\n' "$out" | awk '/Measured steps:/{print $NF}')" \
    "$dev" \
    "$(printf '%s\n' "$out" | awk '/Step loop per stage:/{print $NF}')" \
    "$(printf '%s\n' "$out" | awk '/Main per step:/{print $NF}')"
  if [ "$rc" -ne 0 ]; then
    printf '%s\n' "$out" | tail -n 6 | sed 's/^/    | /'
  fi
}

sweep() { # $1 part  $2 rounds  $3.. orders
  local part=$1 rounds=$2; shift 2
  [ $# -gt 0 ] || return 0
  echo
  echo "===== $part (rounds=$rounds, modes=$EMU_MODES) ====="
  local r p mode conf
  for r in $(seq 1 "$rounds"); do
    for p in "$@"; do
      for mode in $EMU_MODES; do
        conf="$OUTDIR/config/perf_p${p}_${mode}.conf"
        [ -f "$conf" ] || mkconf "$conf" "$p" "$mode" || continue
        run_one "r$(printf %02d "$r") p${p} ${mode}" "$conf" \
          "$OUTDIR/run_logs/perf_r$(printf %02d "$r")_p${p}_${mode}.log"
      done
    done
  done
  echo "${part}_DONE"
}

if [ "$TIMING" = 1 ]; then
  sweep PART_LO "$ROUNDS" $ORDERS_LO
  sweep PART_HI "$ROUNDS_HI" $ORDERS_HI
fi

echo
echo "ALL_DONE  artifacts in $OUTDIR (metadata.txt, validation.txt, config/, run_logs/)"
