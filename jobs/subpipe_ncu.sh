#!/bin/bash
#
# Which SM sub-pipe binds the contraction kernels, and how much of it is DMMA.
#
# reports/a100_report.md 15.3 scales each kernel by one roof -- HBM for the
# DRAM-saturated ones, the FP64 tensor roof for the contraction ones -- and
# reproduces the p=511 ranking inversion on both sides (GB200 0.949 vs 0.950
# measured, A100 1.076 vs 1.065).  The same decomposition transferred to H100
# predicts GEMM_FUSED/FUSED_TC = 1.338 against 1.068 measured, and gets
# GEMM/FUSED_TC the wrong way round (1.812 against 0.895).
#
# The reason is structural, not a bad fit.  Per SM-cycle the three machines
# offer:
#
#            FP64 tensor   FP64 CUDA core   issue   DRAM bytes
#   GB200        127.9          127.9         4        25.21
#   A100         128.1           63.7         4        10.21
#   H100         256.0          128.2         4         9.17
#
# GB200 and A100 have the SAME FP64 tensor throughput per SM-cycle, so on that
# pair "the FP64 tensor roof" and "anything proportional to SM-cycles" are one
# and the same roof and cannot be told apart by any ratio of times.  H100 is
# the only one of the three that separates them, and it separates them by
# exactly the factor 2 that the prediction misses.  A model calibrated on
# GB200+A100 is therefore rank-deficient in precisely the coordinate that
# decides H100.
#
# What breaks the degeneracy without H100 timings is the SM sub-pipe
# decomposition, which is a kernel property measured on the calibration
# machines: FP64 and the tensor core share one issue pipe
# (sm__pipe_shared_cycles_active is documented as "fp64+tensor"), and only the
# DMMA half of its cycles gets cheaper on H100.  So this job measures, per
# kernel:
#
#   * sm__pipe_shared_cycles_active   -> is the fp64+tensor pipe the roof?
#   * sm__pipe_{alu,fma,fp64}_cycles_active -> which other pipe is, if not
#   * sm__inst_executed_pipe_{tensor,fp64,alu,fma,lsu,adu,cbu,uniform,xu}
#                                     -> the per-pipe instruction mix, which is
#                                        what transfers to an unmeasured machine
#   * the roof_bytes set (dram/l1tex/lts/sm throughput, bytes, occupancy)
#                                     -> so the rows join the existing sweeps
#   * l1tex sectors / requests / LSU wavefronts (total, not just the shared
#     ones)                           -> a HARD L1 roof.  l1tex__throughput is a
#     max over sub-pipes, so dividing t_bytes by it gives an implied peak that
#     ranges 15.9-95.7 B/SM-cycle over the kernels of one order (measured, job
#     92507) -- exactly what reports/a100_report.md 10.4 warned about.  The
#     shared-memory wavefronts already collected explain only 0.3-23.6% of that
#     utilisation, so the rest is the global sector/tag path and has to be
#     counted separately before L1 can be a capacity rather than a fitted number.
#   * global-vs-shared LSU instruction counts, global sector counts and the L2
#     hit/miss split -> the SECOND equation.  A kernel that is on no roof is
#     bound by concurrency over latency, T = N_inst * (1 + sum stalls) / (4 W),
#     which is an identity (W = (1 + sum stalls) * IPC, Little's law, verified
#     on job 92904).  Transferring it needs the memory latency in SM cycles,
#     and that cannot be read off sm__inst_executed_pipe_lsu because most LSU
#     instructions are shared-memory ops with short latency: doing so gives
#     1.5-334 SM cycles over the kernels of one machine, i.e. nothing.  The
#     long-latency population is the global loads that miss in L2, so the
#     denominator has to be lg_cmd_read weighted by the L2 miss rate.
#
# All of these exist on gb100, gh100 and ga100 (checked with
# ncu --query-metrics --chip), so the same sweep runs on all three machines.
#
# CORRECTION (2026-09-10, reports/subpipe_transfer_model.md 5N.3): the query
# returning a name is not the same as the counter having a value.  On sm_80
# sm__inst_executed_pipe_lsu_mem_lg_cmd_read.sum comes back "n/a" for every
# kernel, so the long-latency population above is NOT collected on A100.  Use
# l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum there, which is collected on
# all three and is identical kernel by kernel.
#
# ncu is a mechanism instrument here, not a wall-time result (AGENTS.md).
# --clock-control none by default, for the reason reports/a100_report.md 12.2
# gives: percentages taken under a locked clock cannot be lifted to boost.
#
# Machine-independent.  Run through a wrapper:
#   RIKYU:    sbatch jobs/rikyu_subpipe_ncu.sh
#   Wisteria: pjsub  jobs/wisteria_subpipe_ncu.sh
#   TSUBAME:  qsub -g <GROUP> jobs/tsubame_subpipe_ncu.sh   (validation only)
#
# Knobs: same as jobs/l1_roof_ncu.sh (ORDERS, PATHS, NSTEP, NCU_TIMEOUT,
#        REPEATS, CLOCKS, TIME_BUDGET, KERNELS, LAUNCH_SKIP, LAUNCH_COUNT,
#        SCALE_DG_MODULES, SCALE_DG_MODULE_PURGE, OUTDIR).
#
# LAUNCH_SKIP/LAUNCH_COUNT default to 4/24, matching jobs/roof_bytes_ncu.sh so
# that the per-launch medians line up with the sweeps already in the tree.
# KERNELS is widened to the GEMM paths and to the peripheral kernels: the
# narrow regex is what cost reports/a100_report.md 15.2 a whole analysis.

set -u

# Freeze this script before running it.  bash reads a script incrementally, so
# editing it while a job is executing it shifts every later byte offset and the
# job dies with a syntax error in the middle of the sweep -- which is exactly
# what happened to Slurm 94875 (reports/subpipe_transfer_model.md 5S.4), after
# its 24 cells had finished.  AGENTS.md already requires profiling a frozen copy
# of the executable for the same class of reason; the job script needs it too.
if [ -z "${SCALE_DG_FROZEN_SCRIPT:-}" ] && [ -r "$0" ]; then
  frozen_self=$(mktemp "${TMPDIR:-/tmp}/subpipe_ncu.XXXXXX") || exit 1
  cat "$0" > "$frozen_self" || exit 1
  export SCALE_DG_FROZEN_SCRIPT="$frozen_self"
  bash "$frozen_self" "$@"
  rc=$?
  rm -f "$frozen_self"
  exit $rc
fi

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

# Colon- and comma-separated lists are accepted as well as space-separated
# ones, because neither SGE's -v nor TCS's -x can carry a value with a space in
# it, and neither scheduler exports the submitting shell's environment by
# default.  jobs/roof_bytes_ncu.sh has done this since it was written; this
# script had not, so
#     ORDERS="767" qsub -g <GROUP> jobs/tsubame_subpipe_orders.sh
# silently ran the default order list.  The working forms are
#     qsub -g <GROUP> -v ORDERS=767 jobs/tsubame_subpipe_orders.sh
#     pjsub -x ORDERS=7:15:31    jobs/wisteria_subpipe_orders.sh
#     ORDERS="7 15" sbatch jobs/rikyu_subpipe_orders.sh      (Slurm does export)
readonly ORDERS=$(printf '%s' "${ORDERS:-63 255 511}" | tr ':,' '  ')
readonly PATHS=$(printf '%s' "${PATHS:-CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM CUDAFORTRAN_FUSED}" | tr ':,' '  ')
readonly NSTEP=${NSTEP:-4}
# Measured, not guessed: on GB200 (Slurm 82908) all nine invocations finished
# in one minute of wall clock, about 7 s each.  300 s is roughly forty times
# that, which is a bound rather than a budget -- AGENTS.md is explicit that a
# generous profiler timeout only delays the retry, because a bad node yields no
# output at all rather than slow output.  It also makes the job self-limiting:
# nine invocations at 300 s is 45 minutes, inside every short queue's cap.
readonly NCU_TIMEOUT=${NCU_TIMEOUT:-1200}
readonly REPEATS=${REPEATS:-1}
readonly CLOCKS=${CLOCKS:-"none"}
# The per-invocation timeout alone no longer bounds the job once REPEATS and
# CLOCKS multiply the invocation count, so bound the whole sweep instead.  The
# budget is checked before each invocation, which keeps the job inside a
# two-hour queue even if every invocation hangs.
readonly TIME_BUDGET=${TIME_BUDGET:-5400}
readonly T_START=$(date +%s)
readonly JOB_TAG=${JOB_ID:-${SLURM_JOB_ID:-${PJM_JOBID:-manual}}}
readonly OUTDIR=${OUTDIR:-"subpipe_ncu_${JOB_TAG}"}
# flux_all is NOT optional: SCALE_DG_FLUXFUSE defaults to 1, which fuses
# volume_flux_kernel and elembnd_flux_kernel into flux_all_kernel wherever they
# would run back to back, and that name matched none of the alternatives above
# (the capital-K "Kernel" does not match "..._flux_all_kernel_").  Every sweep
# before 2026-09-11 therefore dropped it silently, which is where the 42-54%
# shortfall of the low-order unfused GEMM cells against production came from
# (reports/subpipe_transfer_model.md 5V.1, 5Y).
readonly KERNELS=${KERNELS:-'regex:tendency_|elembnd|volume_flux|flux_all|separable_lift|cutlass|gemm|Kernel'}
readonly LAUNCH_SKIP=${LAUNCH_SKIP:-4}
readonly LAUNCH_COUNT=${LAUNCH_COUNT:-24}
# CutlassMmaShape.  On sm_100 every shape lowers to DMMA.8x8x4 (2/4/8 of
# them for 16x8x4 / 16x8x8 / 16x8x16) while sm_90 issues each as one
# instruction -- reports/sm90_mma_shape_survey.md 6.  Sweeping it on GB200
# is therefore a control: the shared-pipe cycle count must NOT move.
readonly SHAPES=$(printf '%s' "${SHAPES:-8x8x4}" | tr ':,' '  ')

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
  echo "kernels=$KERNELS launch_skip=$LAUNCH_SKIP launch_count=$LAUNCH_COUNT"
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
readonly METRICS_A='gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_fp64.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_adu.sum,sm__inst_executed_pipe_cbu.sum,sm__inst_executed_pipe_uniform.sum,sm__inst_executed_pipe_xu.sum,smsp__inst_executed.avg.per_cycle_active,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts.sum,l1tex__t_sectors.sum,l1tex__t_requests.sum,l1tex__lsuin_requests.sum,l1tex__f_wavefronts.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,sm__inst_executed_pipe_lsu_mem_lg_cmd_read.sum,sm__inst_executed_pipe_lsu_mem_shared_cmd_read.sum,lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers,launch__grid_size,launch__block_size'
# Group B is separate on purpose: it is the stall breakdown, which
# answers "what is the kernel waiting on" rather than "which pipe is
# full".  Splitting it also means an unavailable name in one group
# cannot take the other group's counters down with it.
readonly METRICS_B='gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio'

# Configs first: they do not depend on the repeat or the clock mode.
for p in $ORDERS; do
  for path in $PATHS; do
    for shp in $SHAPES; do
      tag="p${p}_${path#CUDAFORTRAN_}_${shp}"
      conf="$OUTDIR/config/${tag}.conf"
      sed -e "s/DqdtKernel_Type *= *\"[A-Z0-9_]*\"/DqdtKernel_Type = \"$path\"/" \
          "namelists/perf_p${p}_gemm.conf" > "$conf"
      sed -i -e "s/nstep *= *[0-9]*/nstep = $NSTEP/" \
             -e "s/output_interval *= *[0-9]*/output_interval = $NSTEP/" "$conf"
      if grep -q CutlassMmaShape "$conf"; then
        sed -i "s/CutlassMmaShape *= *\"[0-9x]*\"/CutlassMmaShape = \"$shp\"/" "$conf"
      else
        sed -i "s|^/|  CutlassMmaShape = \"$shp\"\n/|" "$conf"
      fi
    done
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
       for shp in $SHAPES; do
        tag="p${p}_${path#CUDAFORTRAN_}_${shp}"
        out="$OUTDIR/ncu/${tag}_${clk}_r$(printf %02d "$r")"
        echo "ncu: $tag clock=$clk r=$r"
        # -s 4 skips the first tendency call, whose caches are cold; -c 8 is
        # enough to cover one full call for every path (p=15 launches one kernel
        # per stage, p=63 and p=255 launch two or three plus the face kernel).
        for grp in a b; do
          case "$grp" in
            a) mset="$METRICS_A" ;;
            b) mset="$METRICS_B" ;;
          esac
          timeout "$NCU_TIMEOUT" ncu --csv \
            --clock-control "$clk" \
            --metrics "$mset" \
            --kernel-name-base demangled --rename-kernels 0 \
            -k "$KERNELS" -s "$LAUNCH_SKIP" -c "$LAUNCH_COUNT" \
            "$EXE" "$OUTDIR/config/${tag}.conf" > "${out}_${grp}.csv" 2> "${out}_${grp}.err"
          echo "  grp=$grp rc=$? lines=$(wc -l < "${out}_${grp}.csv")"
        done
       done
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
