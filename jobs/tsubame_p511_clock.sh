#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dg_h100_p511clk
#$ -j y
#
# H100 half of the p=511 FUSED_TC clock experiment (prediction Q4).  See
# jobs/wisteria_p511_clock.sh for the reasoning; H100's stage MAD on this cell
# is 0.103% against A100's 0.244% and GB200's 0.011%, so the same sign with a
# smaller effect is what Q4 expects.
#
# BEFORE SUBMITTING:
#   module purge && module load nvhpc
#   make CUDA=1 GPUFLAGS=-gpu=cc90 GPUNVCCFLAGS=-arch=sm_90
#   qsub -g <TSUBAME_GROUP> jobs/tsubame_p511_clock.sh
set -u
module purge >/dev/null 2>&1 || true
module load nvhpc
R="${SGE_O_WORKDIR:-$PWD}"
O="$R/p511_clock_${JOB_ID:-manual}"
mkdir -p "$O/config"; cd "$R" || exit 1
echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
# The five paths jobs/tsubame_paths.sh runs at p=511, in its order.  The
# two-path version of this job (SGE 8595368) did NOT reproduce the signature
# it was chasing -- FUSED_TC 0.67% against GEMM_FUSED 0.53%, where the harness
# gives 0.103 against 0.005 -- so the protocol has to match the harness, not
# just the namelist.  With five paths a given path returns to the GPU once a
# minute instead of once every twenty seconds.
PATHS="FUSED_TC FUSED_DFMA GEMM GEMM_CUTE GEMM_FUSED"
for k in $PATHS; do
  sed "s/DqdtKernel_Type *= *\"[A-Z_]*\"/DqdtKernel_Type = \"CUDAFORTRAN_$k\"/" \
    "$R/namelists/perf_p511_gemm.conf" > "$O/config/$k.conf"
done
# clocks_throttle_reasons.active does not exist on the A100 node's driver
# (535.54.03): job 9540286 got two samples and an error.  Probe first, and
# fall back to the four fields every driver here has.
FIELDS=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active
nvidia-smi --query-gpu=$FIELDS --format=csv,noheader -c 1 >/dev/null 2>&1 || \
  FIELDS=timestamp,clocks.sm,power.draw,temperature.gpu
echo "sampler fields: $FIELDS"
nvidia-smi --query-gpu=$FIELDS \
           --format=csv,noheader --loop-ms=100 > "$O/smi.csv" 2>"$O/smi.err" &
SMI=$!
sleep 2
for r in $(seq 1 20); do
  for k in $PATHS; do
    echo "=== RUN r$r $k start=$(date +%s.%N)"
    ./scale-dg_extraction "$O/config/$k.conf"
    echo "=== RUN r$r $k end=$(date +%s.%N)"
  done
done > "$O/ab.log" 2>&1
kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
echo "samples=$(wc -l < "$O/smi.csv")"
echo "ALL_DONE"
