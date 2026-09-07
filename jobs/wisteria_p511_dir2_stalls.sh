#!/bin/bash
#PJM -N dg_a100_dir2_stalls
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# measurement_inventory.md 21.16 left exactly one thing unmeasured: which
# stall grows in the slow processes.  Everything ncu counts is identical --
# instructions to the digit, DRAM bytes to 0.04%, DRAM active cycles to 0.05%,
# occupancy exactly -- and the duration still moves 2.2%, so the answer has to
# be in the stall profile or nowhere.
#
# Predictions are frozen in output/s38/PREDICTIONS.txt, including what it means
# if no single reason carries it.
#
# BEFORE SUBMITTING:
#   module unload nvidia && module load nvidia/25.9
#   make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
#   fill in -g above, then:  pjsub jobs/wisteria_p511_dir2_stalls.sh
set -u
module unload nvidia >/dev/null 2>&1 || true
module load nvidia/25.9
export DEBUGINFOD_URLS=
R="${PJM_O_WORKDIR:-$PWD}"
O="$R/p511_dir2_stalls_${PJM_JOBID:-manual}"
mkdir -p "$O"; cd "$R" || exit 1
echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD)"

CONF=$O/p511.conf
sed 's/DqdtKernel_Type *= *"[A-Z_]*"/DqdtKernel_Type = "CUDAFORTRAN_FUSED_TC"/' \
  "$R/namelists/perf_p511_gemm.conf" > "$CONF"

# Enumerate the stall reasons rather than listing them by hand: section 21.5
# of dfma_register_budget.md records what happens when eight of seventeen go
# missing.
timeout 300 ncu --query-metrics 2>/dev/null > "$O/allmetrics.txt"
grep -oE 'smsp__average_warps_issue_stalled_[a-z_]+_per_issue_active' "$O/allmetrics.txt" \
  | sort -u > "$O/stalls.txt"
echo "stall reasons: $(wc -l < "$O/stalls.txt")"
STALL=$(sed 's/$/.ratio/' "$O/stalls.txt" | paste -sd, -)

M="gpu__time_duration.sum,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed.sum,$STALL"

# Nine launches is three full stages, so DIR2 is seen three times and the
# first captured launch can be dropped -- section 21.16 measured that one to
# be an outlier in three processes out of eight.
for p in 1 2 3 4 5 6 7 8; do
  echo "=== proc $p"
  timeout 900 ncu --csv --clock-control none --metrics "$M" \
    --kernel-name-base demangled --rename-kernels 0 \
    -k regex:tendency_p255_kernel -s 3 -c 9 \
    "$R/scale-dg_extraction" "$CONF" > "$O/S_p$p.csv" 2>"$O/S_p$p.err"
  echo "   exit=$? size=$(stat -c%s "$O/S_p$p.csv" 2>/dev/null)"
done
echo "ALL_DONE"
