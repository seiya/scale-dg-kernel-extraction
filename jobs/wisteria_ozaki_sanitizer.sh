#!/bin/bash
#PJM -N dg_a100_oz2san
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# A100 half of the OZAKI2 crash diagnosis.  See jobs/tsubame_ozaki_sanitizer.sh
# for why: prediction O1 failed, the padding hypothesis is dead, and CUDA's
# asynchronous errors mean "x-deriv" is only where the fault was noticed.
#
# BEFORE SUBMITTING:
#   module unload nvidia && module load nvidia/25.9
#   make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
#   fill in -g above, then:  pjsub jobs/wisteria_ozaki_sanitizer.sh
set -u
# Job 9539840 exited 127 on both configurations: this script had no module
# load, so nvcc was not on PATH, the fallback collapsed to
# "./compute-sanitizer", and nothing ran.  Load the compiler the same way
# jobs/wisteria_paths.sh does, then search rather than guess.
module unload nvidia >/dev/null 2>&1 || true
module load nvidia/25.9
R="${PJM_O_WORKDIR:-$PWD}"
O="$R/oz2_sanitizer_${PJM_JOBID:-manual}"
mkdir -p "$O"; cd "$R" || exit 1
SAN=$(command -v compute-sanitizer || true)
if [ -z "$SAN" ] && command -v nvcc >/dev/null 2>&1; then
  NVBIN=$(dirname "$(command -v nvcc)")
  for c in "$NVBIN/compute-sanitizer" \
           "$NVBIN/../compute-sanitizer/compute-sanitizer" \
           "$NVBIN/../../cuda/bin/compute-sanitizer"; do
    [ -x "$c" ] && SAN="$c" && break
  done
fi
if [ -z "$SAN" ] || [ ! -x "$SAN" ]; then
  echo "ERROR: compute-sanitizer not found. PATH=$PATH" >&2
  echo "  try: find \$(dirname \$(command -v nvcc))/../.. -name compute-sanitizer -type f" >&2
  exit 2
fi
echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD)"
echo "sanitizer=$SAN"; "$SAN" --version 2>&1 | head -2
mk() { sed -e 's/DqdtKernel_Type *= *"[A-Z_]*"/DqdtKernel_Type = "CUDAFORTRAN_GEMM_OZAKI2"/' \
           -e "s/nstep *= *[0-9]*/nstep = 1/" "$2" > "$1"; }
mk "$O/small.conf" "$R/namelists/val_p7_gemm.conf"
mk "$O/full.conf"  "$R/namelists/perf_p7_gemm.conf"
for tag in small full; do
  echo "===== compute-sanitizer memcheck: $tag"
  timeout 1500 "$SAN" --tool memcheck --launch-timeout 120 --print-limit 20 \
    ./scale-dg_extraction "$O/$tag.conf" > "$O/san_$tag.log" 2>&1
  echo "   exit=$?"
  echo "--- first error block ---"
  grep -m1 -A12 'Invalid __global__\|Invalid __shared__\|Invalid __local__\|out-of-bounds' "$O/san_$tag.log" || echo "   (no memcheck error found)"
  tail -3 "$O/san_$tag.log"
done
echo "ALL_DONE"
