#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -N dg_h100_oz2san
#$ -j y
#
# ozaki2_implementation_report.md 8.4 prediction O1 FAILED: padding the INT8
# buffers changed nothing (SGE 8594961 at 3b66356 still gives 93 rc=0 / 51
# failures, identical to the unpadded run).  So the over-read hypothesis is
# wrong and guessing has to stop.
#
# CUDA errors are asynchronous: "ozaki2_dgemm x-deriv" is only where the error
# was NOTICED.  compute-sanitizer names the kernel and the address, which is
# the measurement this needs.
#
# BEFORE SUBMITTING:
#   module purge && module load nvhpc
#   make CUDA=1 GPUFLAGS=-gpu=cc90 GPUNVCCFLAGS=-arch=sm_90
#   qsub -g <TSUBAME_GROUP> jobs/tsubame_ozaki_sanitizer.sh
#
# Two configurations, smallest first.  memcheck costs 10-100x, so Ne=2^3 at
# p=7 is tried first; if it does not reproduce, the production p=7 case runs.
# A small case is legitimate here because this is a crash diagnosis, not a
# performance comparison.
set -u
module purge >/dev/null 2>&1 || true
module load nvhpc
R="${SGE_O_WORKDIR:-$PWD}"
O="$R/oz2_sanitizer_${JOB_ID:-manual}"
mkdir -p "$O"; cd "$R" || exit 1
SAN=$(command -v compute-sanitizer || echo "$(dirname "$(command -v nvcc)")/compute-sanitizer")
echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD)"
echo "sanitizer=$SAN"; "$SAN" --version 2>&1 | head -2

mk() { sed -e 's/DqdtKernel_Type *= *"[A-Z_]*"/DqdtKernel_Type = "CUDAFORTRAN_GEMM_OZAKI2"/' \
           -e "s/nstep *= *[0-9]*/nstep = 1/" "$2" > "$1"; }
mk "$O/small.conf" "$R/namelists/val_p7_gemm.conf"     # Ne=2^3
mk "$O/full.conf"  "$R/namelists/perf_p7_gemm.conf"    # Ne=32^3

for tag in small full; do
  echo "===== compute-sanitizer memcheck: $tag"
  timeout 1500 "$SAN" --tool memcheck --launch-timeout 120 --print-limit 20 \
    ./scale-dg_extraction "$O/$tag.conf" > "$O/san_$tag.log" 2>&1
  echo "   exit=$?"
  echo "--- first error block ---"
  grep -m1 -A12 'Invalid __global__\|Invalid __shared__\|Invalid __local__\|out-of-bounds' "$O/san_$tag.log" || echo "   (no memcheck error found)"
  grep -c 'ERROR SUMMARY' "$O/san_$tag.log" >/dev/null && tail -3 "$O/san_$tag.log"
done
echo "ALL_DONE"
