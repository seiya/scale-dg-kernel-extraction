#!/bin/bash
# Build and run jobs/global_latency_probe.cu.  Machine-independent body; the
# wrappers below set ARCH and the module.  The probe is a plain CUDA run, not a
# profiler, so on RIKYU it may go on the login GPU; TSUBAME and Wisteria need a
# job because their login nodes have no GPU.
#
#   RIKYU:    bash jobs/global_latency_probe.sh          (ARCH=sm_100)
#   TSUBAME:  qsub -g <GROUP> jobs/tsubame_global_latency.sh
#   Wisteria: pjsub jobs/wisteria_global_latency.sh
set -u
ROOT=""
for c in "${SCALE_DG_ROOT:-}" "${SGE_O_WORKDIR:-}" "${PJM_O_WORKDIR:-}" "$PWD"; do
  [ -n "$c" ] && [ -f "$c/jobs/global_latency_probe.cu" ] && { ROOT="$c"; break; }
done
[ -n "$ROOT" ] || { echo "ERROR: checkout not found" >&2; exit 1; }
cd "$ROOT" || exit 1
[ "${SCALE_DG_MODULE_PURGE:-1}" = 1 ] && module purge
for m in ${SCALE_DG_MODULES:-nvhpc}; do module load "$m"; done
ARCH=${ARCH:-sm_100}
OUT=${OUT:-global_latency_${ARCH}}
nvcc -O3 -arch="$ARCH" -o "/tmp/glat_$$" jobs/global_latency_probe.cu || exit 1
{
  date --iso-8601=seconds
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown) arch=$ARCH host=$(hostname)"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.max.sm --format=csv,noheader
  "/tmp/glat_$$"
} | tee "${OUT}.txt"
rm -f "/tmp/glat_$$"
