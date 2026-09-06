#!/bin/bash
#
# The third term the additive roofline is missing.
#
# reports/a100_report.md section 9: fitting t = a*(FLOP/peak) + b*(bytes/BW)
# with machine-independent a,b is exact by construction on two machines and
# fails on the third -- 18.0% RMS over 44 path-order pairs, signs correlated
# with the machine, and some rows wanting a negative compute coefficient.
# The failure is concentrated in the CUDA-core paths (FUSED, FUSED_DFMA),
# which miss by 17-45% on GB200 while the mma paths sit within a few percent
# at low order.  Forcing the CC paths to fit puts H100's effective compute
# rate at 13.9-16.8 TFLOP/s against a nominal 33.5 -- 42-50%, consistent
# across eleven pairs -- which says those paths are not limited by the FP64
# pipe at all, so a fit against the FP64 roof absorbs whatever does limit
# them.  h100_report.md 8.7.3 already named that limiter for one kernel: L1
# latency and occupancy, worth 13-20% with no DRAM traffic involved.
#
# This job measures the candidate third term directly, per kernel:
#   * l1tex throughput as a fraction of ITS peak  -> is L1 the binding roof?
#   * fp64 pipe utilisation                        -> is the FP64 roof idle?
#   * shared-memory wavefronts and bank conflicts  -> where the L1 time goes
#   * occupancy, registers, issue efficiency       -> the 8.7.3 mechanism
# so that a third term proportional to SM count and clock, rather than to
# FP64 FLOPs, can be fitted on measurements instead of assumed.
#
# ncu locks the clocks, so NOTHING here is a wall-time result (AGENTS.md).
# It is used to identify which resource binds; adoption decisions stay on
# interleaved wall-time A/B.
#
# Machine-independent.  Run through a wrapper:
#   RIKYU:    sbatch jobs/rikyu_l1_roof_ncu.sh
#   Wisteria: pjsub  jobs/wisteria_l1_roof_ncu.sh
#
# Knobs: ORDERS (default "15 63 255"), PATHS (default the three fused paths),
#        NSTEP (default 4), NCU_TIMEOUT (default 300 s per invocation),
#        REPEATS (default 1), CLOCKS (default "base"), TIME_BUDGET (default
#        5400 s), SCALE_DG_MODULES, SCALE_DG_MODULE_PURGE, OUTDIR
#
# REPEATS and CLOCKS exist for the retake described in
# reports/a100_report.md section 11.7.  ncu's default --clock-control base
# locks the SM clock, and it locks it to a DIFFERENT FRACTION of boost on each
# machine (0.53x on GB200, 0.77x on A100), which biases any cross-machine
# comparison of which roof binds: dram% is the only utilisation that moves
# with SM clock, so DRAM pressure is understated, and understated more on
# GB200.  Section 11.6 corrects for that analytically and finds the bias too
# small to overturn the conclusion, but leaves six kernels whose top two roofs
# sit within ten points.  Sweeping CLOCKS="base none" with REPEATS=10 settles
# those: every run self-reports its clock through sm__cycles_elapsed, so dram%
# can be regressed on clock instead of merely averaged, which also tests
# section 11.6's linearity assumption.
#
# The defaults reproduce the original single-pass runs (Slurm 82908,
# TCS 9523261): REPEATS=1 CLOCKS=base.

set -u

# Not from $0: SGE and TCS both run a spool copy.  See jobs/tsubame_paths.sh.
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
  exit 1
fi
cd "$ROOT" || exit 1
echo "root=$ROOT"

readonly MODULES=${SCALE_DG_MODULES:-nvhpc}
readonly MODULE_PURGE=${SCALE_DG_MODULE_PURGE:-1}
[ "$MODULE_PURGE" = 1 ] && module purge
for m in $MODULES; do module load "$m"; done

readonly ORDERS=${ORDERS:-"15 63 255"}
readonly PATHS=${PATHS:-"CUDAFORTRAN_FUSED CUDAFORTRAN_FUSED_DFMA CUDAFORTRAN_FUSED_TC"}
readonly NSTEP=${NSTEP:-4}
# Measured, not guessed: on GB200 (Slurm 82908) all nine invocations finished
# in one minute of wall clock, about 7 s each.  300 s is roughly forty times
# that, which is a bound rather than a budget -- AGENTS.md is explicit that a
# generous profiler timeout only delays the retry, because a bad node yields no
# output at all rather than slow output.  It also makes the job self-limiting:
# nine invocations at 300 s is 45 minutes, inside every short queue's cap.
readonly NCU_TIMEOUT=${NCU_TIMEOUT:-300}
readonly REPEATS=${REPEATS:-1}
readonly CLOCKS=${CLOCKS:-"base"}
# The per-invocation timeout alone no longer bounds the job once REPEATS and
# CLOCKS multiply the invocation count, so bound the whole sweep instead.  The
# budget is checked before each invocation, which keeps the job inside a
# two-hour queue even if every invocation hangs.
readonly TIME_BUDGET=${TIME_BUDGET:-5400}
readonly T_START=$(date +%s)
readonly JOB_TAG=${JOB_ID:-${SLURM_JOB_ID:-${PJM_JOBID:-manual}}}
readonly OUTDIR=${OUTDIR:-"l1_roof_ncu_${JOB_TAG}"}
readonly KERNELS='regex:tendency_|pdl_elembnd'

mkdir -p "$OUTDIR/config" "$OUTDIR/ncu" "$OUTDIR/logs" || exit 1
if [ ! -x ./scale-dg_extraction ]; then
  echo "ERROR: ./scale-dg_extraction missing; build first." >&2
  exit 1
fi
# Freeze: a concurrent make must not relink under the running profiler.
cp -p ./scale-dg_extraction "$OUTDIR/scale-dg_extraction.frozen"
EXE="$(cd "$OUTDIR" && pwd)/scale-dg_extraction.frozen"

{
  echo "date=$(date --iso-8601=seconds)"
  echo "job=${JOB_TAG} host=$(hostname)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "orders=$ORDERS nstep=$NSTEP repeats=$REPEATS clocks=$CLOCKS"
  echo "ncu_timeout=$NCU_TIMEOUT time_budget=$TIME_BUDGET"
  echo "paths=$PATHS"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm --format=csv,noheader
  nvcc --version 2>&1 | tail -n 2
  ncu --version 2>&1 | head -n 4
  echo "modules:"; module list 2>&1
} > "$OUTDIR/metadata.txt"
cat "$OUTDIR/metadata.txt"

# One line on purpose: a backslash-continued single-quoted string keeps the
# backslash and the newline IN the value, and ncu then reports every metric as
# n/a (jobs/split_deriv_ncu.sh learnt this the hard way on job 81637).
readonly METRICS='gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,l1tex__t_bytes.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers,launch__grid_size,launch__block_size,sm__inst_executed.sum,smsp__inst_executed.avg.per_cycle_active,sm__cycles_elapsed.avg.per_second'

# Configs first: they do not depend on the repeat or the clock mode.
for p in $ORDERS; do
  for path in $PATHS; do
    tag="p${p}_${path#CUDAFORTRAN_}"
    conf="$OUTDIR/config/${tag}.conf"
    sed -e "s/DqdtKernel_Type *= *\"[A-Z0-9_]*\"/DqdtKernel_Type = \"$path\"/" \
        "namelists/perf_p${p}_gemm.conf" > "$conf"
    sed -i -e "s/nstep *= *[0-9]*/nstep = $NSTEP/" \
           -e "s/output_interval *= *[0-9]*/output_interval = $NSTEP/" "$conf"
  done
done

# Repeats outermost so that a drift in clocks or in the machine's state hits
# every configuration equally, exactly as the path sweep interleaves rounds.
budget_left() {
  local now; now=$(date +%s)
  [ $((now - T_START)) -lt "$TIME_BUDGET" ]
}
for r in $(seq 1 "$REPEATS"); do
  for clk in $CLOCKS; do
    for p in $ORDERS; do
      for path in $PATHS; do
        if ! budget_left; then
          echo "TIME_BUDGET ${TIME_BUDGET}s reached; stopping before r$r $clk p$p $path"
          break 4
        fi
        tag="p${p}_${path#CUDAFORTRAN_}"
        out="$OUTDIR/ncu/${tag}_${clk}_r$(printf %02d "$r")"
        echo "ncu: $tag clock=$clk r=$r"
        # -s 4 skips the first tendency call, whose caches are cold; -c 8 is
        # enough to cover one full call for every path (p=15 launches one kernel
        # per stage, p=63 and p=255 launch two or three plus the face kernel).
        timeout "$NCU_TIMEOUT" ncu --csv \
          --clock-control "$clk" \
          --metrics "$METRICS" \
          --kernel-name-base demangled --rename-kernels 0 \
          -k "$KERNELS" -s 4 -c 8 \
          "$EXE" "$OUTDIR/config/${tag}.conf" > "${out}.csv" 2> "${out}.err"
        echo "  rc=$? lines=$(wc -l < "${out}.csv")"
      done
    done
  done
done

echo
echo "===== which roof binds, per kernel ====="
python3 - "$OUTDIR" <<'PY'
import csv, glob, os, re, statistics as st, sys
outdir = sys.argv[1]
M = {
  'sm__throughput.avg.pct_of_peak_sustained_elapsed': 'sm',
  'l1tex__throughput.avg.pct_of_peak_sustained_elapsed': 'l1',
  'lts__throughput.avg.pct_of_peak_sustained_elapsed': 'l2',
  'dram__throughput.avg.pct_of_peak_sustained_elapsed': 'dram',
  'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active': 'fp64',
  'sm__warps_active.avg.pct_of_peak_sustained_active': 'occ',
  'launch__registers_per_thread': 'regs',
  'sm__cycles_elapsed.avg.per_second': 'hz',
}
ORDER = ['sm', 'l1', 'l2', 'dram', 'fp64']

def num(x):
    try:
        return float(x.replace(',', ''))
    except (AttributeError, ValueError):
        return None

def short(n):
    return n.split('(')[0].replace('void ', '')[:26]

# samples[(tag, kernel, clock)] -> list of dicts, one per profiled launch
samples = {}
for path in sorted(glob.glob(os.path.join(outdir, 'ncu', '*.csv'))):
    base = os.path.basename(path)[:-4]
    m = re.match(r'(.+)_(base|none|hz)_r(\d+)$', base)
    tag, clk = (m.group(1), m.group(2)) if m else (base, 'base')
    try:
        lines = [l for l in open(path, newline='') if l.strip()]
        start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    except (OSError, StopIteration):
        continue
    per = {}
    for r in csv.DictReader(lines[start:]):
        name = M.get(r.get('Metric Name'))
        if name:
            per.setdefault((r['ID'], short(r['Kernel Name'])), {})[name] = num(r['Metric Value'])
    for (i, k), v in per.items():
        samples.setdefault((tag, k, clk), []).append(v)

def med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return st.median(v) if v else None

def fmt(x, w=5):
    return ('%*.1f' % (w, x)) if x is not None else ('%*s' % (w, '-'))

for (tag, k, clk) in sorted(samples):
    rows = samples[(tag, k, clk)]
    vals = {n: med(rows, n) for n in ORDER}
    top = max((v, n) for n, v in vals.items() if v is not None)
    hz = med(rows, 'hz')
    spread = ''
    d = [r['dram'] for r in rows if r.get('dram') is not None]
    if len(d) > 1:
        spread = '  dram %.1f-%.1f' % (min(d), max(d))
    print('%-22s %-28s %-4s n=%-3d %s' % (tag, k, clk, len(rows),
          ' '.join('%s=%s' % (n, fmt(vals[n])) for n in ORDER)) +
          '  occ=%s regs=%s clk=%s GHz  -> %s %.1f%s' % (
              fmt(med(rows, 'occ')), fmt(med(rows, 'regs')),
              ('%.3f' % (hz / 1e9)) if hz else '-', top[1], top[0], spread))

# dram% is the only utilisation that scales with SM clock (its roof is set by
# the memory clock, not the SM clock).  With more than one clock in the sweep,
# regress it and report the slope: reports/a100_report.md 11.6 assumes exactly
# this linearity when it extrapolates the locked measurement to boost.
print()
print('===== dram%% against SM clock (tests 11.6 linearity) =====')
by_kernel = {}
for (tag, k, clk), rows in samples.items():
    for r in rows:
        if r.get('dram') is not None and r.get('hz'):
            by_kernel.setdefault((tag, k), []).append((r['hz'] / 1e9, r['dram']))
for key in sorted(by_kernel):
    pts = by_kernel[key]
    xs = sorted(set(round(x, 3) for x, _ in pts))
    if len(pts) < 4 or len(xs) < 2:
        print('%-22s %-28s n=%-3d clocks=%s  (too few distinct clocks to fit)' % (
            key[0], key[1], len(pts), xs))
        continue
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx if sxx else float('nan')
    # Proportionality would put the line through the origin: dram% = c * clock.
    ratio = (slope * mx / my) if my else float('nan')
    print('%-22s %-28s n=%-3d clocks=%.3f..%.3f  slope=%.1f %%/GHz  '
          'slope*clk/dram=%.2f (1.00 = proportional)' % (
              key[0], key[1], n, min(xs), max(xs), slope, ratio))
PY

echo
echo "ALL_DONE  artifacts in $OUTDIR"
