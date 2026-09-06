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
#        NSTEP (default 4), SCALE_DG_MODULES, SCALE_DG_MODULE_PURGE, OUTDIR

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
  echo "orders=$ORDERS nstep=$NSTEP"
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

for p in $ORDERS; do
  for path in $PATHS; do
    tag="p${p}_${path#CUDAFORTRAN_}"
    conf="$OUTDIR/config/${tag}.conf"
    sed -e "s/DqdtKernel_Type *= *\"[A-Z0-9_]*\"/DqdtKernel_Type = \"$path\"/" \
        "namelists/perf_p${p}_gemm.conf" > "$conf"
    sed -i -e "s/nstep *= *[0-9]*/nstep = $NSTEP/" \
           -e "s/output_interval *= *[0-9]*/output_interval = $NSTEP/" "$conf"
    echo "ncu: $tag"
    # -s 4 skips the first tendency call, whose caches are cold; -c 8 is enough
    # to cover one full call for every path (p=15 launches one kernel per stage,
    # p=63 and p=255 launch two or three plus the face kernel).
    timeout 900 ncu --csv \
      --metrics "$METRICS" \
      --kernel-name-base demangled --rename-kernels 0 \
      -k "$KERNELS" -s 4 -c 8 \
      "$EXE" "$conf" > "$OUTDIR/ncu/${tag}.csv" 2> "$OUTDIR/logs/${tag}.err"
    echo "  rc=$? lines=$(wc -l < "$OUTDIR/ncu/${tag}.csv")"
  done
done

echo
echo "===== which roof binds, per kernel ====="
python3 - "$OUTDIR" <<'PY'
import csv, glob, os, sys
outdir = sys.argv[1]
WANT = {
  'gpu__time_duration.sum': 'dur',
  'sm__throughput.avg.pct_of_peak_sustained_elapsed': 'sm%',
  'l1tex__throughput.avg.pct_of_peak_sustained_elapsed': 'l1%',
  'lts__throughput.avg.pct_of_peak_sustained_elapsed': 'l2%',
  'dram__throughput.avg.pct_of_peak_sustained_elapsed': 'dram%',
  'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active': 'fp64%',
  'sm__warps_active.avg.pct_of_peak_sustained_active': 'occ%',
  'launch__registers_per_thread': 'regs',
  'smsp__inst_executed.avg.per_cycle_active': 'ipc',
}
for path in sorted(glob.glob(os.path.join(outdir, 'ncu', '*.csv'))):
    tag = os.path.basename(path)[:-4]
    rows = {}
    try:
        with open(path, newline='') as fh:
            # ncu --csv prefixes the table with a banner; find the header row
            lines = [l for l in fh if l.strip()]
        start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
        if start is None:
            print('--- %s: no csv table (see logs/)' % tag); continue
        for r in csv.DictReader(lines[start:]):
            k = r.get('Kernel Name', '')[:44]
            m = r.get('Metric Name')
            if m in WANT:
                rows.setdefault(k, {})[WANT[m]] = r.get('Metric Value')
    except Exception as exc:
        print('--- %s: %s' % (tag, exc)); continue
    print('--- %s' % tag)
    for k, v in rows.items():
        print('  %-46s sm=%-7s l1=%-7s l2=%-7s dram=%-7s fp64=%-7s occ=%-7s regs=%-5s ipc=%s' % (
            k, v.get('sm%','-'), v.get('l1%','-'), v.get('l2%','-'), v.get('dram%','-'),
            v.get('fp64%','-'), v.get('occ%','-'), v.get('regs','-'), v.get('ipc','-')))
PY

echo
echo "ALL_DONE  artifacts in $OUTDIR"
