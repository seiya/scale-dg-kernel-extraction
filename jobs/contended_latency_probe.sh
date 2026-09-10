#!/bin/bash
# Build and run jobs/contended_latency_probe.cu.  See jobs/global_latency_probe.sh.
#   RIKYU:    bash jobs/contended_latency_probe.sh
#   TSUBAME:  qsub -g <GROUP> jobs/tsubame_contended_latency.sh
#   Wisteria: pjsub jobs/wisteria_contended_latency.sh
set -u
ROOT=""
for c in "${SCALE_DG_ROOT:-}" "${SGE_O_WORKDIR:-}" "${PJM_O_WORKDIR:-}" "$PWD"; do
  [ -n "$c" ] && [ -f "$c/jobs/contended_latency_probe.cu" ] && { ROOT="$c"; break; }
done
[ -n "$ROOT" ] || { echo "ERROR: checkout not found" >&2; exit 1; }
cd "$ROOT" || exit 1
[ "${SCALE_DG_MODULE_PURGE:-1}" = 1 ] && module purge
for m in ${SCALE_DG_MODULES:-nvhpc}; do module load "$m"; done
ARCH=${ARCH:-sm_100}; OUT=${OUT:-contended_latency_${ARCH}}
nvcc -O3 -arch="$ARCH" -o "/tmp/clat_$$" jobs/contended_latency_probe.cu -lnvidia-ml || exit 1
{
  date --iso-8601=seconds
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown) arch=$ARCH host=$(hostname)"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.max.sm,clocks.max.mem --format=csv,noheader
  for r in 1 2 3; do echo "== round $r"; "/tmp/clat_$$"; done
} | tee "${OUT}.txt"
rm -f "/tmp/clat_$$"
