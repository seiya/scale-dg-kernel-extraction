#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=03:00:00
#$ -N dg_roof_bytes
#$ -j y
#SBATCH --job-name=dg_roof_bytes
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#
# Measured DRAM bytes, and which roof binds, for the fastest paths.
#
# Two questions, one job, because both want the same per-launch counters over
# one whole tendency call:
#
#  1. reports/h100_report.md 8.9.2 retracted "both machines are pinned to the
#     FP64 roof at p=767".  Priced against the roof its kernels actually use
#     (DMMA, 66.9 TF on H100), the H100 winner sits at 46.8% of compute and
#     18.1% of bandwidth -- neither roof -- and what does bind it is unmeasured.
#     That is the report's only open item.
#
#  2. Every DRAM figure in the published tables is unique bytes / time: an
#     algorithmic minimum, not traffic.  The two differ by a lot -- changing
#     the p=255 z tile alone moved dram__bytes_read by 44.5% with the
#     arithmetic unchanged (reports/p255_gap_study.md) -- so the reports cannot
#     currently separate algorithmic from measured arithmetic intensity.
#
# So: no -k filter (cuBLAS kernel names are not predictable and the surrounding
# kernels are part of the call -- reports/a100_report.md 15.2 lost a whole
# analysis to a -k regex that omitted volume_flux), per-launch bytes and
# utilisations, and enough launches to cover one call and detect its period.
#
# --clock-control none by default.  AGENTS.md: ncu's default lock is a
# different fraction of boost on each machine (0.53x GB200, 0.77x A100), and
# when the question is which roof binds in production the answer is to re-take
# at boost rather than correct on paper.  Byte COUNTS are clock-invariant and
# would be valid either way; the utilisation percentages are not.
#
# Nothing here is a wall-time result.  Mechanism only (AGENTS.md).
#
# TSUBAME:  qsub -g <GROUP> -v SHAPE=16x8x4 jobs/roof_bytes_ncu.sh
# RIKYU:    sbatch jobs/rikyu_roof_bytes_ncu.sh
# Wisteria: pjsub  jobs/wisteria_roof_bytes_ncu.sh
#
# Knobs: ORDERS, PATHS_<p> (per-order path list), SHAPE, CLOCK, NSTEP,
#        SKIP_CALLS, COUNT, NCU_TIMEOUT, REPLAY, TIME_BUDGET, OUTDIR,
#        SCALE_DG_ROOT, SCALE_DG_MODULES, SCALE_DG_MODULE_PURGE

set -u

# $0 is a spool copy under SGE and TCS; find the checkout the same way the
# rest of jobs/ does.
ROOT=""
for candidate in "${SCALE_DG_ROOT:-}" "${SGE_O_WORKDIR:-}" "${PJM_O_WORKDIR:-}" "$PWD" \
                 "$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; do
  [ -n "$candidate" ] || continue
  [ -f "$candidate/namelists/perf_p7_gemm.conf" ] || continue
  [ -f "$candidate/Makefile" ] || continue
  [ -w "$candidate" ] || continue
  ROOT="$candidate"; break
done
[ -n "$ROOT" ] || { echo "ERROR: no writable checkout found" >&2; exit 1; }
cd "$ROOT" || exit 1
echo "root=$ROOT"

readonly MODULES=${SCALE_DG_MODULES:-nvhpc}
[ "${SCALE_DG_MODULE_PURGE:-1}" = 1 ] && module purge
for m in $MODULES; do module load "$m"; done
export DEBUGINFOD_URLS=

readonly ORDERS=${ORDERS:-"7 63 255 767"}
readonly SHAPE=${SHAPE:-8x8x4}
readonly CLOCK=${CLOCK:-none}
readonly NSTEP=${NSTEP:-4}
readonly SKIP_CALLS=${SKIP_CALLS:-4}
readonly COUNT=${COUNT:-24}
# p=767 runs a single volume GEMM for tens of milliseconds and ncu replays it
# once per metric pass, so bound each invocation at a few times the measured
# cost of the p=511 run rather than generously (AGENTS.md: on a bad node the
# profiler emits nothing at all, so a long bound only delays the retry).
readonly NCU_TIMEOUT=${NCU_TIMEOUT:-1200}
# ncu's default kernel replay saves and restores whatever the kernel writes, so
# it needs a device-side buffer on top of the payload.  That is free on GB200
# (189 GiB) and H100 (94 GiB) at these orders, but p=575 already sits at 81% of
# an A100's 40 GiB (jobs/wisteria_paths.sh), where the buffer can push it over.
# REPLAY=application re-runs the whole application once per pass instead, which
# costs time and not memory; with nstep=4 that trade is cheap.  Default kernel,
# so the GB200 and H100 runs are unchanged.
readonly REPLAY=${REPLAY:-kernel}
readonly TIME_BUDGET=${TIME_BUDGET:-6000}
readonly T_START=$(date +%s)
readonly JOB_TAG=${JOB_ID:-${SLURM_JOB_ID:-${PJM_JOBID:-manual}}}
readonly OUTDIR=${OUTDIR:-"roof_bytes_ncu_${JOB_TAG}"}

# Which paths exist at which order: the fused kernels stop at p=511, and the
# published winners are FUSED_TC up to p=255 and the GEMM family above it
# (reports/h100_report.md 8.1).  GEMM and GEMM_FUSED are carried at every
# order so that the measured-bytes column has the same two paths throughout.
paths_for() {
  local p=$1 var
  var="PATHS_$p"
  if [ -n "${!var:-}" ]; then printf '%s' "${!var}"; return; fi
  case "$p" in
    511|575|767|1023) echo "CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_FUSED" ;;
    *) echo "CUDAFORTRAN_FUSED CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_FUSED" ;;
  esac
}

mkdir -p "$OUTDIR/config" "$OUTDIR/ncu" "$OUTDIR/logs" || exit 1
[ -x ./scale-dg_extraction ] || { echo "ERROR: build first" >&2; exit 1; }
# Freeze: a concurrent make must not relink under the running profiler.
cp -p ./scale-dg_extraction "$OUTDIR/scale-dg_extraction.frozen"
EXE="$(cd "$OUTDIR" && pwd)/scale-dg_extraction.frozen"

{
  echo "date=$(date --iso-8601=seconds)"
  echo "job=${JOB_TAG} host=$(hostname)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "orders=$ORDERS shape=$SHAPE clock=$CLOCK nstep=$NSTEP"
  echo "skip_calls=$SKIP_CALLS count=$COUNT ncu_timeout=$NCU_TIMEOUT replay=$REPLAY"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm --format=csv,noheader
  nvfortran --version 2>&1 | head -n 2
  ncu --version 2>&1 | head -n 4
  echo "modules:"; module list 2>&1
} > "$OUTDIR/metadata.txt"
cat "$OUTDIR/metadata.txt"

readonly METRICS='gpu__time_duration.sum,dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__grid_size,launch__block_size,sm__inst_executed.sum,smsp__inst_executed.avg.per_cycle_active,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,sm__cycles_elapsed.avg.per_second'

budget_left() { [ $(( $(date +%s) - T_START )) -lt "$TIME_BUDGET" ]; }

for p in $ORDERS; do
  for path in $(paths_for "$p"); do
    budget_left || { echo "TIME_BUDGET reached before p$p $path"; break 2; }
    tag="p${p}_${path#CUDAFORTRAN_}"
    conf="$OUTDIR/config/${tag}.conf"
    sed -e "s/DqdtKernel_Type *= *\"[A-Z0-9_]*\"/DqdtKernel_Type = \"$path\"/" \
        "namelists/perf_p${p}_gemm.conf" > "$conf"
    if grep -q CutlassMmaShape "$conf"; then
      sed -i "s/CutlassMmaShape *= *\"[0-9x]*\"/CutlassMmaShape = \"$SHAPE\"/" "$conf"
    else
      sed -i "s|^/|  CutlassMmaShape = \"$SHAPE\"\n/|" "$conf"
    fi
    sed -i -e "s/nstep *= *[0-9]*/nstep = $NSTEP/" \
           -e "s/output_interval *= *[0-9]*/output_interval = $NSTEP/" "$conf"

    # Skip whole tendency calls, not launches: the paths launch different
    # numbers of kernels per call, so a fixed launch skip would start each
    # path at a different place in its call.  Six launches per call is an
    # upper bound for every path here, so SKIP_CALLS*6 lands past the cold
    # ones for all of them.
    skip=$(( SKIP_CALLS * 6 ))
    echo "ncu: $tag (skip=$skip count=$COUNT clock=$CLOCK)"
    timeout "$NCU_TIMEOUT" ncu --csv --clock-control "$CLOCK" --metrics "$METRICS" \
      --kernel-name-base demangled --rename-kernels 0 \
      --replay-mode "$REPLAY" \
      -s "$skip" -c "$COUNT" \
      "$EXE" "$conf" > "$OUTDIR/ncu/${tag}.csv" 2> "$OUTDIR/logs/${tag}.err"
    echo "  rc=$? lines=$(wc -l < "$OUTDIR/ncu/${tag}.csv")"
  done
done

echo
echo "===== per launch: bytes, roofs, stalls ====="
python3 - "$OUTDIR" <<'PY'
import csv, glob, os, re, sys
outdir = sys.argv[1]
COLS = [("gpu__time_duration.sum", "dur_us"),
        ("dram__bytes.sum", "dram_B"),
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM%"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM%"),
        ("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "L1%"),
        ("lts__throughput.avg.pct_of_peak_sustained_elapsed", "L2%"),
        ("sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active", "fp64%"),
        ("sm__warps_active.avg.pct_of_peak_sustained_active", "occ%"),
        ("launch__registers_per_thread", "regs"),
        ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "long"),
        ("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", "short"),
        ("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", "mio"),
        ("smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "math")]

def num(x):
    try: return float(str(x).replace(",", ""))
    except ValueError: return None

def short(n):
    n = re.sub(r"<.*", "", n or "?").split("(")[0].replace("void ", "")
    return n[-40:]

for path in sorted(glob.glob(os.path.join(outdir, "ncu", "*.csv"))):
    tag = os.path.basename(path)[:-4]
    lines = [l for l in open(path, newline="") if l.strip()]
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    except StopIteration:
        print("== %s == (no rows)" % tag); continue
    launches = {}
    for r in csv.DictReader(lines[start:]):
        launches.setdefault((int(r["ID"]), short(r["Kernel Name"])), {})[r["Metric Name"]] = r["Metric Value"]
    ids = sorted(launches)
    if not ids:
        print("== %s == (no rows)" % tag); continue
    # One tendency call = the launches up to the next repeat of the first
    # kernel name.  Every path launches a fixed sequence per call.
    first = ids[0][1]
    period = next((i for i, (k, n) in enumerate(ids) if i and n == first), len(ids))
    print("== %s ==  (call = %d launches)" % (tag, period))
    print("  %-40s %s" % ("kernel", " ".join("%10s" % s for _, s in COLS)))
    tot_dur = tot_B = 0.0
    for (kid, name) in ids[:period]:
        m = launches[(kid, name)]
        print("  %-40s %s" % (name, " ".join("%10s" % m.get(k, "-") for k, _ in COLS)))
        d, b = num(m.get("gpu__time_duration.sum")), num(m.get("dram__bytes.sum"))
        tot_dur += d or 0.0; tot_B += b or 0.0
    print("  CALL TOTAL: dur=%.1f us  dram=%.3f GB  -> %.3f TB/s" % (
        tot_dur, tot_B / 1e9, (tot_B / (tot_dur * 1e-6)) / 1e12 if tot_dur else 0.0))
    print()
PY

echo "ROOF_BYTES_NCU_DONE  artifacts in $OUTDIR"
