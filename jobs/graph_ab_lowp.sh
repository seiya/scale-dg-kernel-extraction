#!/bin/bash
#
# CUDA-graph on/off A/B at the production orders (p=7, 15), all six paper
# paths, interleaved in one job.  Shared body; one thin wrapper per scheduler
# (rikyu_graph_ab_lowp.sh / tsubame_graph_ab_lowp.sh / wisteria_graph_ab_lowp.sh).
#
# Why: the paper times every path with UseCudaGraph = .false. and cites five
# non-interleaved GB200 development-log pairs (514853f) for what the graph
# would save.  A reviewer asked whether the p <= 15 rankings survive with the
# graph on.  The rankings closest to the graph's per-path difference are
# FUSED_TC/FUSED at p=7 (H100 1.007) and the fastest-path rows at p=7, 15, so
# this job measures off and on side by side, six rounds, and the ratio tables
# are recomputed from graph_ab_lowp.csv on the paper side.
#
# Method (same as jobs/tsubame_paths.sh, which the published table uses):
#   * namelists/perf_p{7,15}_<path>.conf as committed; only UseCudaGraph is
#     substituted (and CutlassMmaShape for the H100 preferred-shape specs).
#     Ne, dt, nstep and the warm-up are untouched.
#   * round-robin: for round / for path / for graph in false true, so the two
#     arms of one pair are adjacent in time.
#   * metrics: "Main per step" and "Step loop per stage".  The tendency CUDA
#     events are off in graph mode ("not measured (graph)"), so there is no
#     device column; the paper's p <= 15 rankings use the stage wall time.
#   * validation: one graph-off / graph-on pair per (order, path) dumps the
#     final q (SCALE_DG_DUMP_Q) and the two are compared point by point.
#
# Knobs (environment):
#   ROUNDS=6            interleaved rounds
#   ORDERS="7 15"       orders
#   SCALE_DG_PATHS      path specs (<kernel>[@<mma_shape>]); default below
#   SCALE_DG_SHAPE_SPECS  extra specs appended (H100: the two @16x8x4; A100: "")
#   VALIDATE=1          final-q agreement, graph off vs on, one pair per cell
#   OUTDIR=...          output tree (default graph_ab_lowp_<jobid>)
#   EXE=...             executable (default: a frozen copy of ./scale-dg_extraction)
set -u

ROOT=""
for candidate in "${SCALE_DG_ROOT:-}" "${SGE_O_WORKDIR:-}" "${PJM_O_WORKDIR:-}" \
                 "${SLURM_SUBMIT_DIR:-}" "$PWD" \
                 "$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; do
  [ -n "$candidate" ] || continue
  [ -f "$candidate/namelists/perf_p7_gemm.conf" ] || continue
  [ -f "$candidate/Makefile" ] || continue
  [ -w "$candidate" ] || continue
  ROOT="$candidate"; break
done
if [ -z "$ROOT" ]; then
  echo "ERROR: cannot find a writable scale-dg-kernel-extraction checkout." >&2
  exit 1
fi
cd "$ROOT" || exit 1
echo "root=$ROOT"

if [ -n "${SGE_O_WORKDIR:-}" ]; then
  readonly MODULES=${SCALE_DG_MODULES:-nvhpc/26.1_cuda13.1}
else
  readonly MODULES=${SCALE_DG_MODULES:-nvhpc}
fi
readonly MODULE_PURGE=${SCALE_DG_MODULE_PURGE:-1}
[ "$MODULE_PURGE" = 1 ] && module purge
for m in $MODULES; do module load "$m"; done

readonly ROUNDS=${ROUNDS:-6}
readonly ORDERS=$(printf '%s' "${ORDERS-7 15}" | tr ':,' '  ')
readonly VALIDATE=${VALIDATE:-1}
readonly JOB_TAG=${JOB_ID:-${SLURM_JOB_ID:-${PJM_JOBID:-manual}}}
readonly MACHINE=${SCALE_DG_MACHINE:-h100}
readonly OUTDIR=${OUTDIR:-"graph_ab_lowp_${JOB_TAG}"}
readonly RESOURCE_LINE=${SCALE_DG_RESOURCE:-"gpu_1=1"}
readonly N="$ROOT/namelists"
# The six paper paths.  The H100 wrapper appends the two CUTLASS paths at the
# sm_90 preferred shape; the A100 wrapper appends nothing (sm_80 refuses it).
readonly SHAPE_SPECS=${SCALE_DG_SHAPE_SPECS-" CUDAFORTRAN_GEMM_CUTE@16x8x4 CUDAFORTRAN_GEMM_FUSED@16x8x4"}
readonly PATHS=$(printf '%s' "${SCALE_DG_PATHS-CUDAFORTRAN_FUSED CUDAFORTRAN_FUSED_TC CUDAFORTRAN_FUSED_DFMA CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED${SHAPE_SPECS}}" | tr ':,' '  ')

mkdir -p "$OUTDIR/config" "$OUTDIR/run_logs" "$OUTDIR/scratch" || exit 1

if [ -n "${EXE:-}" ]; then
  EXE="$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")"
else
  [ -x ./scale-dg_extraction ] || { echo "ERROR: ./scale-dg_extraction missing" >&2; exit 1; }
  cp -p ./scale-dg_extraction "$OUTDIR/scale-dg_extraction.$MACHINE"
  EXE="$(cd "$OUTDIR" && pwd)/scale-dg_extraction.$MACHINE"
fi
readonly EXE

{
  echo "date=$(date --iso-8601=seconds)"
  echo "job_id=$JOB_TAG"
  echo "host=$(hostname)"
  echo "cwd=$(pwd)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "exe=$EXE"
  echo "exe_sha256=$(sha256sum "$EXE" | cut -d' ' -f1)"
  echo "rounds=$ROUNDS orders=$ORDERS validate=$VALIDATE"
  echo "paths=$PATHS"
  echo "machine=$MACHINE"
  echo "resource=$RESOURCE_LINE"
  echo
  echo "gpu:"
  nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.max.sm \
    --format=csv,noheader
  echo "gpu_clocks_power:"
  nvidia-smi --query-gpu=clocks.applications.graphics,clocks.sm,power.limit,power.max_limit,temperature.gpu,clocks_event_reasons.active \
    --format=csv 2>&1
  echo
  echo "nvfortran_version:"; nvfortran --version 2>&1 | head -n 4
  echo "module_list:"; module list 2>&1
  echo "git_status:"; git status --short 2>/dev/null | head -n 40
} > "$OUTDIR/metadata.txt"
cat "$OUTDIR/metadata.txt"

# Namelist: the committed per-path perf namelist, UseCudaGraph substituted.
make_config() { # $1 out  $2 order  $3 spec  $4 graph(false|true)
  local out=$1 order=$2 spec=$3 graph=$4
  local kernel=${spec%%@*} shape=""
  case "$spec" in *@*) shape=${spec#*@} ;; esac
  local suffix
  case "$kernel" in
    CUDAFORTRAN_FUSED)       suffix=fused ;;
    CUDAFORTRAN_FUSED_TC)    suffix=fused_tc ;;
    CUDAFORTRAN_FUSED_DFMA)  suffix=fused_dfma ;;
    CUDAFORTRAN_GEMM)        suffix=gemm ;;
    CUDAFORTRAN_GEMM_CUTE)   suffix=gemm_cute ;;
    CUDAFORTRAN_GEMM_FUSED)  suffix=gemm_fused ;;
    *) echo "ERROR: no namelist for $kernel" >&2; return 1 ;;
  esac
  local template="$N/perf_p${order}_${suffix}.conf"
  [ -f "$template" ] || { echo "ERROR: no template $template" >&2; return 1; }
  grep -q "DqdtKernel_Type *= *\"$kernel\"" "$template" || { echo "ERROR: $template is not $kernel" >&2; return 1; }
  sed -e "s/UseCudaGraph *= *\.[a-z]*\./UseCudaGraph = .$graph./" "$template" > "$out" || return 1
  grep -q "UseCudaGraph = .$graph." "$out" || { echo "ERROR: UseCudaGraph not set in $out" >&2; return 1; }
  if [ -n "$shape" ]; then
    if grep -q "CutlassMmaShape" "$out"; then
      sed -i -e "s/CutlassMmaShape *= *\"[0-9x]*\"/CutlassMmaShape = \"$shape\"/" "$out"
    else
      sed -i -e "s|^/|  CutlassMmaShape = \"$shape\"\n/|" "$out"
    fi
  fi
}

label_of() { echo "$1" | tr '@' '_' | tr -d '"'; }

cmp_dump() { # $1 a  $2 b  -- streaming, as in tsubame_paths.sh
  python3 - "$1" "$2" <<'PY'
import sys, math
worst = ref = sd2 = sx2 = 0.0; n = 0; ndiff = 0
with open(sys.argv[1]) as a, open(sys.argv[2]) as b:
    for la, lb in zip(a, b):
        x = float(la); y = float(lb); d = abs(x - y)
        if d > worst: worst = d
        if d > 0: ndiff += 1
        if abs(x) > ref: ref = abs(x)
        sd2 += d * d; sx2 += x * x; n += 1
rel = worst / ref if ref else 0.0
l2 = math.sqrt(sd2) / math.sqrt(sx2) if sx2 else 0.0
print("n=%d ndiff=%d max|diff|=%.3e ref_max=%.3e rel=%.3e relL2=%.3e" % (n, ndiff, worst, ref, rel, l2))
PY
}

if [ "$VALIDATE" = 1 ]; then
  echo
  echo "===== VALIDATION (final q, graph off vs on, one pair per cell) ====="
  : > "$OUTDIR/validation.txt"
  for p in $ORDERS; do
    for spec in $PATHS; do
      lbl=$(label_of "$spec")
      ok=1
      for g in false true; do
        conf="$OUTDIR/config/val_p${p}_${lbl}_${g}.conf"
        make_config "$conf" "$p" "$spec" "$g" || { ok=0; break; }
        if ! env SCALE_DG_DUMP_Q="$OUTDIR/scratch/q_${g}" "$EXE" "$conf" \
            > "$OUTDIR/run_logs/val_p${p}_${lbl}_${g}.log" 2>&1; then
          echo "p=$p $spec graph=$g RUN FAILED" | tee -a "$OUTDIR/validation.txt"
          tail -n 3 "$OUTDIR/run_logs/val_p${p}_${lbl}_${g}.log"; ok=0; break
        fi
      done
      if [ "$ok" = 1 ]; then
        printf 'p=%-4s %-34s ' "$p" "$spec" | tee -a "$OUTDIR/validation.txt"
        cmp_dump "$OUTDIR/scratch/q_false" "$OUTDIR/scratch/q_true" | tee -a "$OUTDIR/validation.txt"
      fi
      rm -f "$OUTDIR/scratch/q_false" "$OUTDIR/scratch/q_true"
    done
  done
  echo "VALIDATION_DONE"
fi

run_one() { # $1 tag  $2 conf  $3 run-log
  local out rc
  out=$(timeout 600 "$EXE" "$2" 2>&1); rc=$?
  printf '%s\n' "$out" > "$3"
  printf "%-56s rc=%-3s steps=%-6s stage=%-12s main=%s\n" "$1" "$rc" \
    "$(printf '%s\n' "$out" | awk '/Measured steps:/{print $NF}')" \
    "$(printf '%s\n' "$out" | awk '/Step loop per stage:/{print $NF}')" \
    "$(printf '%s\n' "$out" | awk '/Main per step:/{print $NF}')"
  [ "$rc" -ne 0 ] && printf '%s\n' "$out" | tail -n 6 | sed 's/^/    | /'
}

echo
echo "===== TIMING (rounds=$ROUNDS) ====="
for r in $(seq 1 "$ROUNDS"); do
  for p in $ORDERS; do
    for spec in $PATHS; do
      lbl=$(label_of "$spec")
      for g in false true; do
        conf="$OUTDIR/config/perf_p${p}_${lbl}_${g}.conf"
        [ -f "$conf" ] || make_config "$conf" "$p" "$spec" "$g" || continue
        run_one "r$(printf %02d "$r") p${p} ${spec} graph=${g}" "$conf" \
          "$OUTDIR/run_logs/perf_r$(printf %02d "$r")_p${p}_${lbl}_${g}.log"
      done
    done
  done
done
echo "TIMING_DONE"
echo "ALL_DONE  artifacts in $OUTDIR (metadata.txt, validation.txt, config/, run_logs/)"
