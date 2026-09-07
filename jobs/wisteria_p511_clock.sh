#!/bin/bash
#PJM -N dg_a100_p511clk
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# measurement_inventory.md 19.7, last item: the mechanism behind the p=511
# FUSED_TC spread on A100.  Section 21.1 showed it is reproducible and not
# node interference; the noise is also graded GB200 < H100 < A100, which is
# the order of power headroom.  This tests the clock directly instead of
# reaching for a profiler: sample the SM clock and power while the interleaved
# rounds run, then correlate each run's time with the clock during it.
# Predictions: output/s36/PREDICTIONS.txt.
#
# BEFORE SUBMITTING: build for A100 as jobs/wisteria_paths.sh describes,
# fill in -g above, then:  pjsub jobs/wisteria_p511_clock.sh
set -u
module unload nvidia >/dev/null 2>&1 || true
module load nvidia/25.9
R="${PJM_O_WORKDIR:-$PWD}"
O="$R/p511_clock_${PJM_JOBID:-manual}"
mkdir -p "$O/config"; cd "$R" || exit 1
echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv

for k in FUSED_TC GEMM_FUSED; do
  sed "s/DqdtKernel_Type *= *\"[A-Z_]*\"/DqdtKernel_Type = \"CUDAFORTRAN_$k\"/" \
    "$R/namelists/perf_p511_gemm.conf" > "$O/config/$k.conf"
done

# background sampler: timestamp, SM clock, power, temperature, every 100 ms
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
  for k in FUSED_TC GEMM_FUSED; do
    echo "=== RUN r$r $k start=$(date +%s.%N)"
    ./scale-dg_extraction "$O/config/$k.conf"
    echo "=== RUN r$r $k end=$(date +%s.%N)"
  done
done > "$O/ab.log" 2>&1

kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
echo "samples=$(wc -l < "$O/smi.csv")"
echo "ALL_DONE"
