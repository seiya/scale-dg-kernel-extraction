#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=08:00:00
#$ -N dg_h100_paths
#$ -j y
#
# TSUBAME 4 (H100) cross-path re-measurement of the current tree.
#
# Replaces jobs/tsubame_ch7.sh, which measured commit 9eed9e5 (2026-08-26) at
# p=7 and p=255 only and is the source of reports/h100_report.md section 3.
# Everything that report measured has since moved (GEMM-family surrounding
# kernel fusion, the volume-GEMM side stream, the p=255 FUSED_TC rewrite that
# took the p=255 crown from GEMM_FUSED, the p=511 fused paths), so this job
# re-takes the whole path x order table on H100 with the same inputs the
# GB200 table uses.
#
# Method, and why it is this and not nsys:
#   * The GB200 summary table (reports/README.md "最新結果のまとめ") is built
#     from the application's own CUDA-event device timers with the paths
#     interleaved round-robin, not from one nsys run per path.  This job does
#     the same so the two columns are comparable.  nsys/ncu on H100 belongs in
#     a separate job (mechanism, not the published time).
#   * Every path at one order runs from ONE namelist -- namelists/perf_p*_gemm.conf
#     with DqdtKernel_Type substituted -- so Ne, dt, nstep and UseCudaGraph are
#     identical across paths by construction (AGENTS.md, Profiling section).
#   * Rounds are interleaved so that a drift in clocks or in neighbours hits
#     every path equally.  Report the median over rounds, never one run.
#
# Build on TSUBAME before submitting (cc90 / sm_90):
#   module purge
#   module load nvhpc
#   make clean
#   make CUDA=1 GPUFLAGS=-gpu=cc90 GPUNVCCFLAGS=-arch=sm_90
#   qsub -g <TSUBAME_GROUP> jobs/tsubame_paths.sh
#
# Submit FROM the checkout: the job writes its output tree into the working
# copy and finds it through SGE_O_WORKDIR.  From anywhere else, name it:
#   qsub -g <TSUBAME_GROUP> -v SCALE_DG_ROOT=/path/to/checkout \
#        /path/to/checkout/jobs/tsubame_paths.sh
#
# gpu_1=1 is one complete H100.  Do NOT use gpu_h / node_o: those are 1/2-GPU
# MIG instances and every number here would be off the published axis.
#
# Knobs (environment):
#   ORDERS_A/B/C=.. orders each part sweeps (empty string skips the part)
#   ROUNDS=6        interleaved rounds for p=7..255   (PART A)
#   ROUNDS_HI=6     interleaved rounds for p=511      (PART B)
#   ROUNDS_XHI=3    interleaved rounds for p=575/767  (PART C)
#   VALIDATE=1      point-varying-coefficient validation before any timing
#   VAL_ORDERS=...  orders to validate (default "7 15 31 63 127 255")
#   TIMING=1        run the timing parts
#   OUTDIR=...     output directory (default tsubame_paths_$JOB_ID)
#   EXE=...         executable to measure (default a frozen copy, see below)

set -u

#-----------------------------------------------------------------------------
# Locate the checkout.
#
# NOT from $0: SGE runs a COPY of this script out of the execution host's spool
# directory, so "$(dirname "$0")/.." is the spool's parent.  cd into it
# succeeds -- it exists and is readable -- and the job then dies on the first
# mkdir with "Permission denied", which is what job 8567835 did.  SGE_O_WORKDIR
# is the directory qsub was run from and is the reliable answer; SCALE_DG_ROOT
# overrides it, PWD covers a run outside the batch system, and the $0 form is
# kept last for that case too.  Whichever candidate wins has to look like this
# repository AND be writable, because the job writes its output tree into it.
#-----------------------------------------------------------------------------
ROOT=""
for candidate in "${SCALE_DG_ROOT:-}" "${SGE_O_WORKDIR:-}" "${PJM_O_WORKDIR:-}" "$PWD" \
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
  echo "  SCALE_DG_ROOT=${SCALE_DG_ROOT:-<unset>}" >&2
  echo "  SGE_O_WORKDIR=${SGE_O_WORKDIR:-<unset>}" >&2
  echo "  PJM_O_WORKDIR=${PJM_O_WORKDIR:-<unset>}" >&2
  echo "  PWD=$PWD" >&2
  echo "  dirname(\$0)/..=$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || echo '<none>')" >&2
  echo "Submit from the checkout, or pass SCALE_DG_ROOT=/path/to/checkout." >&2
  exit 1
fi
cd "$ROOT" || exit 1
echo "root=$ROOT"

# The module names differ per machine (TSUBAME/RIKYU: nvhpc; Wisteria
# Aquarius: nvidia/25.9 plus gcc/12.2.0).  Defaults keep TSUBAME and RIKYU
# exactly as they were.
readonly MODULES=${SCALE_DG_MODULES:-nvhpc}
readonly MODULE_PURGE=${SCALE_DG_MODULE_PURGE:-1}
[ "$MODULE_PURGE" = 1 ] && module purge
for m in $MODULES; do module load "$m"; done

readonly ROUNDS=${ROUNDS:-6}
readonly ROUNDS_HI=${ROUNDS_HI:-6}
readonly ROUNDS_XHI=${ROUNDS_XHI:-3}
readonly VALIDATE=${VALIDATE:-1}
readonly VAL_ORDERS=${VAL_ORDERS:-"7 15 31 63 127 255"}
readonly TIMING=${TIMING:-1}
readonly ORDERS_A=${ORDERS_A-"7 15 31 63 127 255"}
readonly ORDERS_B=${ORDERS_B-"511"}
readonly ORDERS_C=${ORDERS_C-"575 767"}
# JOB_ID is SGE (TSUBAME), SLURM_JOB_ID is Slurm (RIKYU): the same script runs
# on both, which is what makes the two columns of the table one measurement.
# PJM_JOBID is Fujitsu TCS (Wisteria).
readonly JOB_TAG=${JOB_ID:-${SLURM_JOB_ID:-${PJM_JOBID:-manual}}}
# Short machine tag: names the frozen executable and, by default, the output
# tree.  h100 keeps the existing TSUBAME/RIKYU names unchanged.
readonly MACHINE=${SCALE_DG_MACHINE:-h100}
readonly OUTDIR=${OUTDIR:-"tsubame_paths_${JOB_TAG}"}
# Hardware roofs printed into metadata.txt, so a report never has to guess
# which peaks a run was scored against.  Defaults are H100.
readonly PEAK_FP64_CUDA=${SCALE_DG_PEAK_FP64_CUDA:-33.5}
readonly PEAK_FP64_TENSOR=${SCALE_DG_PEAK_FP64_TENSOR:-66.9}
readonly PEAK_HBM=${SCALE_DG_PEAK_HBM:-2.39587}
readonly RESOURCE_LINE=${SCALE_DG_RESOURCE:-"gpu_1=1"}
readonly N="$ROOT/namelists"

mkdir -p "$OUTDIR/config" "$OUTDIR/run_logs" "$OUTDIR/scratch" || exit 1

# Measure a frozen copy: a concurrent make must not relink under the job.
if [ -n "${EXE:-}" ]; then
  EXE="$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")"
else
  if [ ! -x ./scale-dg_extraction ]; then
    echo "ERROR: ./scale-dg_extraction is missing; build for cc90 first (see header)." >&2
    exit 1
  fi
  cp -p ./scale-dg_extraction "$OUTDIR/scale-dg_extraction.$MACHINE"
  # Absolute, and correct whether OUTDIR is relative to ROOT or absolute.
  EXE="$(cd "$OUTDIR" && pwd)/scale-dg_extraction.$MACHINE"
fi
readonly EXE

{
  echo "date=$(date --iso-8601=seconds)"
  echo "job_id=${JOB_ID:-unknown}"
  echo "host=$(hostname)"
  echo "cwd=$(pwd)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "exe=$EXE"
  echo "rounds=$ROUNDS rounds_hi=$ROUNDS_HI rounds_xhi=$ROUNDS_XHI"
  echo "validate=$VALIDATE val_orders=$VAL_ORDERS timing=$TIMING"
  echo "machine=$MACHINE"
  echo "resource=$RESOURCE_LINE"
  echo "FP64_CUDA_peak_TFLOPS=$PEAK_FP64_CUDA"
  echo "FP64_Tensor_peak_TFLOPS=$PEAK_FP64_TENSOR"
  echo "HBM_peak_TBps=$PEAK_HBM"
  echo
  echo "gpu:"
  nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.max.sm \
    --format=csv,noheader
  echo
  echo "nvfortran_version:"; nvfortran --version 2>&1 | head -n 4
  echo
  echo "nvcc_version:"; nvcc --version 2>&1 | tail -n 2
  echo
  echo "gcc_version:"; gcc --version 2>&1 | head -n 1
  echo
  echo "module_list:"; module list 2>&1
  echo
  echo "git_status:"; git status --short 2>/dev/null | head -n 40
} > "$OUTDIR/metadata.txt"
cat "$OUTDIR/metadata.txt"

#-----------------------------------------------------------------------------
# Config generation.  One template per order (the GEMM perf namelist), with
# DqdtKernel_Type and, where asked, CutlassMmaShape substituted.  Nothing else
# is touched, which is what keeps the comparison one-axis.
#-----------------------------------------------------------------------------
# path spec: <kernel_type>[@<mma_shape>]
make_config() { # $1 out  $2 order  $3 spec  [$4 nstep override]
  local out=$1 order=$2 spec=$3 nstep_override=${4:-}
  local kernel=${spec%%@*} shape=""
  case "$spec" in *@*) shape=${spec#*@} ;; esac
  local template="$N/perf_p${order}_gemm.conf"
  if [ ! -f "$template" ]; then
    echo "ERROR: no template $template" >&2
    return 1
  fi
  sed -e "s/DqdtKernel_Type *= *\"[A-Z0-9_]*\"/DqdtKernel_Type = \"$kernel\"/" \
      "$template" > "$out" || return 1
  if [ -n "$shape" ]; then
    if grep -q "CutlassMmaShape" "$out"; then
      sed -i -e "s/CutlassMmaShape *= *\"[0-9x]*\"/CutlassMmaShape = \"$shape\"/" "$out"
    else
      sed -i -e "s|^/|  CutlassMmaShape = \"$shape\"\n/|" "$out"
    fi
  fi
}

label_of() { # tsubame-safe file/label token for a path spec
  echo "$1" | tr '@' '_' | tr -d '"'
}

#-----------------------------------------------------------------------------
# Validation.  A new machine and a new build are not covered by the GB200
# record: the CUTLASS and cuBLAS kernels that actually run are different SASS.
# Compare the complete owned dqdt with point-varying u/v/w/Escale/normal_fn.
# Reference: CUDAFORTRAN_SPLIT where it exists (p<=127), CUDAFORTRAN_GEMM above.
#-----------------------------------------------------------------------------
cmp_dump() { # $1 ref  $2 run   -- streams, so a 3 GB dump does not need 3 GB of RAM
  python3 - "$1" "$2" <<'PY'
import sys
worst = 0.0
ref = 0.0
n = 0
with open(sys.argv[1]) as a, open(sys.argv[2]) as b:
    for la, lb in zip(a, b):
        x = float(la); y = float(lb)
        d = abs(x - y)
        if d > worst: worst = d
        if abs(x) > ref: ref = abs(x)
        n += 1
rel = worst / ref if ref else 0.0
print("n=%d max|diff|=%.3e ref_max=%.3e rel=%.3e" % (n, worst, ref, rel))
PY
}

paths_for_order() { # echo the path specs that exist at this order
  #
  # CutlassMmaShape (the "@shape" specs) is the axis that decided the H100
  # ranking last time: 16x8x4 is one SASS instruction on sm_90 and took the
  # p=255 volume GEMM down 23.5%, which handed GEMM_FUSED the crown back
  # (reports/h100_report.md section 4).  On GB200 ptxas expands it into
  # DMMA.8x8x4 and it is worth 0.03%, so the tree's default stays 8x8x4.
  #
  # Both CUTLASS paths take it at EVERY order as of the instantiation of
  # 2026-09-04.  Before that the knob was only half connected: the x volume
  # GEMM's order-specialized tile hard-coded 8x8x4 up to Nq = 256, so a
  # non-default shape moved y and z and left x behind in both paths, and the
  # fused x carrier (Nq = 8, 16, 32) and the Nq = 64 z carrier refused the
  # shape outright.  Validated on GB200 at p = 7..255, both paths, shapes
  # 8x8x4 / 16x8x4 / 16x8x8, full owned dqdt against SPLIT (p<=127) and GEMM
  # (p=255) with point-varying coefficients; the default shape is bit-identical
  # to the pre-change executable.
  #
  # 16x8x8 is not swept here -- it lost on H100 last time and spilled the x
  # GEMM's registers.  Sweeping shapes is jobs/tsubame_mma_shape.sh's job; this
  # job carries 16x8x4 because without it the H100 CUTLASS column would be
  # reported at a shape H100 does not want.
  # Empty on a machine without the sm_90 f64 MMA instructions (A100): the
  # tree now refuses shapes 1-3 there with error stop rather than trapping
  # inside the kernel (reports/a100_prediction.md section 7.2), so sweeping
  # them would only produce failed runs.
  local shapes=${SCALE_DG_SHAPE_SPECS-" CUDAFORTRAN_GEMM_CUTE@16x8x4 CUDAFORTRAN_GEMM_FUSED@16x8x4"}
  case $1 in
    7|15|31|63|127)
      echo "CUDAFORTRAN_SPLIT CUDAFORTRAN_FUSED CUDAFORTRAN_FUSED_TC CUDAFORTRAN_FUSED_DFMA CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED${shapes}" ;;
    255)
      # No SPLIT at p=255 (AGENTS.md Validation); GEMM is the reference.
      echo "CUDAFORTRAN_FUSED CUDAFORTRAN_FUSED_TC CUDAFORTRAN_FUSED_DFMA CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED${shapes}" ;;
    511)
      # No CUDA-core FUSED at p>=511 (p511_gap_study.md 14.6 / 15.7).
      echo "CUDAFORTRAN_FUSED_TC CUDAFORTRAN_FUSED_DFMA CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED${shapes}" ;;
    575|767)
      echo "CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED${shapes}" ;;
    *) echo "" ;;
  esac
}

if [ "$VALIDATE" = 1 ]; then
  echo
  echo "===== VALIDATION (point-varying coefficients, full owned dqdt) ====="
  : > "$OUTDIR/validation.txt"
  for p in $VAL_ORDERS; do
    if [ "$p" -le 127 ]; then ref_kernel=CUDAFORTRAN_SPLIT; else ref_kernel=CUDAFORTRAN_GEMM; fi
    ref_conf="$OUTDIR/config/val_p${p}_ref.conf"
    make_config "$ref_conf" "$p" "$ref_kernel" || continue
    # One step is enough: the dump is taken at the first tendency evaluation.
    sed -i -e "s/nstep *= *[0-9]*/nstep = 1/" -e "s/output_interval *= *[0-9]*/output_interval = 1/" "$ref_conf"
    ref_dump="$OUTDIR/scratch/p${p}.ref"
    echo "--- p=$p  reference=$ref_kernel"
    if ! env SCALE_DG_VARYING_COEFF=1 SCALE_DG_DUMP_DQDT="$ref_dump" \
        "$EXE" "$ref_conf" > "$OUTDIR/run_logs/val_p${p}_ref.log" 2>&1; then
      echo "  REFERENCE RUN FAILED (see run_logs/val_p${p}_ref.log)" | tee -a "$OUTDIR/validation.txt"
      tail -n 5 "$OUTDIR/run_logs/val_p${p}_ref.log"
      rm -f "$ref_dump"
      continue
    fi
    for spec in $(paths_for_order "$p"); do
      [ "${spec%%@*}" = "$ref_kernel" ] && continue
      lbl=$(label_of "$spec")
      conf="$OUTDIR/config/val_p${p}_${lbl}.conf"
      make_config "$conf" "$p" "$spec" || continue
      sed -i -e "s/nstep *= *[0-9]*/nstep = 1/" -e "s/output_interval *= *[0-9]*/output_interval = 1/" "$conf"
      dump="$OUTDIR/scratch/p${p}.run"
      printf 'p=%-4s %-34s ' "$p" "$spec" | tee -a "$OUTDIR/validation.txt"
      if env SCALE_DG_VARYING_COEFF=1 SCALE_DG_DUMP_DQDT="$dump" \
          "$EXE" "$conf" > "$OUTDIR/run_logs/val_p${p}_${lbl}.log" 2>&1; then
        cmp_dump "$ref_dump" "$dump" | tee -a "$OUTDIR/validation.txt"
      else
        echo "RUN FAILED" | tee -a "$OUTDIR/validation.txt"
        tail -n 3 "$OUTDIR/run_logs/val_p${p}_${lbl}.log"
      fi
      rm -f "$dump"
    done
    rm -f "$ref_dump"
  done
  echo "VALIDATION_DONE"
fi

#-----------------------------------------------------------------------------
# Timing.  One line per (round, order, path); post-process by taking the
# median of each (order, path) column.
#
# The application prints, per run:
#   Measured steps:            steps that entered the timed region
#   CUDA device <something>:   CUDA-event device time for the tendency
#   Step loop per stage:       wall time per RK stage
#   Main per step:             end-to-end wall time per step
# The published table uses the device timer; Main is kept as the wall-time
# cross-check that AGENTS.md asks for (device event vs end-to-end).
#-----------------------------------------------------------------------------
run_one() { # $1 tag  $2 conf  $3 run-log
  local out rc dev
  out=$(timeout 1800 "$EXE" "$2" 2>&1); rc=$?
  printf '%s\n' "$out" > "$3"
  # CUDAFORTRAN_SPLIT prints four device timers (volume flux / derivative /
  # surface lift / assembly) where the fused and GEMM paths print one, so the
  # device column is the sum of whatever "CUDA device" lines the path emits.
  # It is NOT comparable across paths on its own -- SPLIT additionally runs
  # elembnd outside these timers.  The cross-path axis is "main" (end-to-end
  # per step), which is the axis the GB200 summary table is bold on; "stage"
  # is the wall time per RK stage.  Per-path device detail stays in the log.
  dev=$(printf '%s\n' "$out" | awk '/CUDA device/{s+=$NF} END{if (s>0) printf "%.5E", s; else printf "-"}')
  printf "%-46s rc=%-3s steps=%-6s devsum=%-12s stage=%-12s main=%s\n" "$1" "$rc" \
    "$(printf '%s\n' "$out" | awk '/Measured steps:/{print $NF}')" \
    "$dev" \
    "$(printf '%s\n' "$out" | awk '/Step loop per stage:/{print $NF}')" \
    "$(printf '%s\n' "$out" | awk '/Main per step:/{print $NF}')"
  if [ "$rc" -ne 0 ]; then
    printf '%s\n' "$out" | tail -n 6 | sed 's/^/    | /'
  fi
}

sweep() { # $1 part-name  $2 rounds  $3.. orders
  local part=$1 rounds=$2; shift 2
  echo
  echo "===== $part (rounds=$rounds) ====="
  local r p spec lbl conf
  for r in $(seq 1 "$rounds"); do
    for p in "$@"; do
      for spec in $(paths_for_order "$p"); do
        lbl=$(label_of "$spec")
        conf="$OUTDIR/config/perf_p${p}_${lbl}.conf"
        [ -f "$conf" ] || make_config "$conf" "$p" "$spec" || continue
        run_one "r$(printf %02d "$r") p${p} ${spec}" "$conf" \
          "$OUTDIR/run_logs/perf_r$(printf %02d "$r")_p${p}_${lbl}.log"
      done
    done
  done
  echo "${part}_DONE"
}

if [ "$TIMING" = 1 ]; then
  # PART A: the six orders the paper's main table is built on.
  sweep PART_A "$ROUNDS" $ORDERS_A

  # PART B: p=511.  Five production paths since the FUSED_TC/DFMA templating
  # (p511_gap_study.md 15); GB200 has GEMM_FUSED fastest by +5.46%.
  [ -n "$ORDERS_B" ] && sweep PART_B "$ROUNDS_HI" $ORDERS_B

  # PART C: p=575 and p=767, GEMM family only.
  #
  # p=767 needs about 78 GiB with the allocator allowance (estimate_device_memory.py)
  # against 94 GB of H100, so it is expected to be tight and may OOM; that is a
  # result, not a job failure -- rc != 0 is printed and the sweep continues.
  # p=1023 is NOT in this job: 180 GiB does not fit on an H100 at all.
  [ -n "$ORDERS_C" ] && sweep PART_C "$ROUNDS_XHI" $ORDERS_C
fi

echo
echo "ALL_DONE  artifacts in $OUTDIR (metadata.txt, validation.txt, config/, run_logs/)"
