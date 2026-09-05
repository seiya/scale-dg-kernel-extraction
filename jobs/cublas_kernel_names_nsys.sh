#!/bin/bash
#
# Which kernels does cuBLAS actually run for the FP64 volume GEMMs?
#
# reports/h100_report.md 8.7.2 found that on GB200 cuBLAS dispatches
# cutlass_80_tensorop_d884gemm -- a CUTLASS-generated sm_80 kernel, the same
# tile as ours, matching even in register count -- so the library axis
# (GEMM vs GEMM_CUTE) is degenerate there and measures nothing.  On H100
# cuBLAS runs a hand-written sm_90 xmma (tensor16x8x8) and beats us 1.20-1.45x
# on instruction count.
#
# That leaves one question open, and it is prediction 3 of
# reports/a100_prediction.md: is the GB200 degeneracy a property of cuBLAS
# reaching for CUTLASS whenever it has nothing better, or is it specific to
# Blackwell, where cuBLAS ships no native FP64 GEMM at all (commit 56c3154)?
# A100 settles it, because sm_80 is that kernel's HOME architecture.
#
#   cuBLAS on A100 emits cutlass_80_tensorop_d884gemm  -> the degeneracy is
#       general; 8.7.2's conclusion stands on two machines.
#   cuBLAS on A100 emits a hand-written kernel         -> the GB200 degeneracy
#       is Blackwell-specific; withdraw the generalisation in 8.7.2.
#
# This is decided by KERNEL NAME, not by time: the timing comparison is
# contaminated because H100 has a real library gap that any fit would absorb.
#
# Machine-independent.  Run it anywhere via a wrapper:
#   Wisteria: pjsub jobs/wisteria_cublas_nsys.sh
#   RIKYU:    sbatch jobs/rikyu_cublas_nsys.sh
#   TSUBAME:  qsub -g <GROUP> jobs/cublas_kernel_names_nsys.sh
#
# Knobs: ORDERS (default "63 255"), NSTEP (default 10),
#        SCALE_DG_MODULES, SCALE_DG_MODULE_PURGE, OUTDIR

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
  echo "  SCALE_DG_ROOT=${SCALE_DG_ROOT:-<unset>} SGE_O_WORKDIR=${SGE_O_WORKDIR:-<unset>}" >&2
  echo "  PJM_O_WORKDIR=${PJM_O_WORKDIR:-<unset>} PWD=$PWD" >&2
  exit 1
fi
cd "$ROOT" || exit 1
echo "root=$ROOT"

readonly MODULES=${SCALE_DG_MODULES:-nvhpc}
readonly MODULE_PURGE=${SCALE_DG_MODULE_PURGE:-1}
[ "$MODULE_PURGE" = 1 ] && module purge
for m in $MODULES; do module load "$m"; done

# Both of these are required or symbol resolution never finishes (AGENTS.md);
# --resolve-symbols=false is on the command line below.
export DEBUGINFOD_URLS=

readonly ORDERS=${ORDERS:-"63 255"}
readonly NSTEP=${NSTEP:-10}
readonly JOB_TAG=${JOB_ID:-${SLURM_JOB_ID:-${PJM_JOBID:-manual}}}
readonly OUTDIR=${OUTDIR:-"cublas_names_nsys_${JOB_TAG}"}

mkdir -p "$OUTDIR/config" "$OUTDIR/nsys" "$OUTDIR/logs" || exit 1
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
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm --format=csv,noheader
  nvcc --version 2>&1 | tail -n 2
  nsys --version 2>&1 | head -n 2
  echo "modules:"; module list 2>&1
} > "$OUTDIR/metadata.txt"
cat "$OUTDIR/metadata.txt"

for p in $ORDERS; do
  for kind in GEMM GEMM_CUTE; do
    tag="p${p}_${kind}"
    conf="$OUTDIR/config/${tag}.conf"
    sed -e "s/DqdtKernel_Type *= *\"[A-Z0-9_]*\"/DqdtKernel_Type = \"CUDAFORTRAN_${kind}\"/" \
        "namelists/perf_p${p}_gemm.conf" > "$conf"
    sed -i -e "s/nstep *= *[0-9]*/nstep = $NSTEP/" \
           -e "s/output_interval *= *[0-9]*/output_interval = $NSTEP/" "$conf"
    # nsys cannot profile the CUDA graph path: it hangs and the report has no
    # GPU trace (AGENTS.md).  The kernels and their order are identical.
    if grep -q "UseCudaGraph" "$conf"; then
      sed -i -e "s/UseCudaGraph *= *\.[a-z]*\./UseCudaGraph = .false./" "$conf"
    else
      sed -i -e "s|^/|  UseCudaGraph = .false.\n/|" "$conf"
    fi
    prefix="$OUTDIR/nsys/${tag}"
    echo "nsys: $tag"
    # timeout from the known duration, not generously: on a bad node the
    # profiler produces nothing rather than running slowly (AGENTS.md).
    timeout 600 nsys profile --trace=cuda --sample=none --cpuctxsw=none \
      --resolve-symbols=false --force-overwrite=true \
      --output="$prefix" "$EXE" "$conf" > "$OUTDIR/logs/${tag}.log" 2>&1
    echo "  profile rc=$?"
    timeout 600 nsys stats --report=cuda_gpu_kern_sum --format=csv \
      --force-export=true --output="$prefix" "${prefix}.nsys-rep" \
      >> "$OUTDIR/logs/${tag}.log" 2>&1
    echo "  stats rc=$?"
  done
done

echo
echo "===== kernels by total GPU time ====="
python3 - "$OUTDIR" <<'PY'
import csv, glob, os, re, sys

outdir = sys.argv[1]
GEMMISH = re.compile(r"gemm|d884|dgemm|xmma|tensorop|Kernel<cutlass", re.I)
verdict = {}

for path in sorted(glob.glob(os.path.join(outdir, "nsys", "*_cuda_gpu_kern_sum.csv"))):
    tag = os.path.basename(path).replace("_cuda_gpu_kern_sum.csv", "")
    rows = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            name = row.get("Name") or ""
            pct = row.get("Time (%)") or row.get("Time(%)") or ""
            inst = row.get("Instances") or ""
            try:
                pct = float(pct)
            except ValueError:
                pct = 0.0
            rows.append((pct, inst, name))
    rows.sort(reverse=True)
    print()
    print("--- %s" % tag)
    for pct, inst, name in rows[:12]:
        mark = "  <== GEMM" if GEMMISH.search(name) else ""
        print("  %6.2f%%  n=%-6s %s%s" % (pct, inst, name[:110], mark))
    gemms = [n for _, _, n in rows if GEMMISH.search(n)]
    verdict[tag] = gemms

print()
print("===== prediction 3 =====")
for tag, gemms in sorted(verdict.items()):
    if not tag.endswith("_GEMM"):
        continue
    cutlass80 = [n for n in gemms if "cutlass_80" in n or "d884gemm" in n]
    print("%s: cuBLAS volume-GEMM kernels = %s" % (tag, gemms[:4] or "<none found>"))
    if cutlass80:
        print("   -> CUTLASS-generated sm_80 kernel present: %s" % cutlass80[0][:100])
        print("      Prediction 3 HOLDS here: the library axis is degenerate,")
        print("      and h100_report.md 8.7.2 generalises beyond Blackwell.")
    elif gemms:
        print("   -> NOT a CUTLASS sm_80 kernel.  Prediction 3 FAILS:")
        print("      the GB200 degeneracy is Blackwell-specific and 8.7.2's")
        print("      generalisation has to be withdrawn.")
    else:
        print("   -> no GEMM-looking kernel found; check %s/logs and the CSV." % outdir)
print()
print("Compare with the matching *_GEMM_CUTE tags: those are our own CUTLASS")
print("kernels, and the question is whether cuBLAS is running the same thing.")
PY

echo
echo "ALL_DONE  artifacts in $OUTDIR"
