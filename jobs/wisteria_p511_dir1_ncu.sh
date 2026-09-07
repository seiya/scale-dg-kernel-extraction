#!/bin/bash
#PJM -N dg_a100_p511_dir1
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# measurement_inventory.md 21.13 narrowed the p=511 variance on A100 to one
# launch: tendency_p255_kernel DIR1 moved 2.84% between two processes while
# DIR0 and DIR2 held to 0.2%.  This is the measurement that decides whether
# that survives eight processes, and what differs when it does.
#
# The GB200 control has already run (Slurm 85814, section 21.15): there all
# three directions sit at 0.08-0.22% across eight processes, so DIR1 is not
# special on that machine, and the two metric passes below are known to be
# accepted by ncu -- pass B's .avg/.max/.min rollups were validated there.
#
# BEFORE SUBMITTING:
#   module unload nvidia && module load nvidia/25.9
#   make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
#   fill in -g above, then:  pjsub jobs/wisteria_p511_dir1_ncu.sh
set -u
module unload nvidia >/dev/null 2>&1 || true
module load nvidia/25.9
export DEBUGINFOD_URLS=
R="${PJM_O_WORKDIR:-$PWD}"
O="$R/p511_dir1_${PJM_JOBID:-manual}"
mkdir -p "$O"; cd "$R" || exit 1
echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD)"
CONF=$O/p511.conf
sed 's/DqdtKernel_Type *= *"[A-Z_]*"/DqdtKernel_Type = "CUDAFORTRAN_FUSED_TC"/' \
  $R/namelists/perf_p511_gemm.conf > $CONF

MA="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,smsp__inst_executed.sum,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active"

MB="gpu__time_duration.sum,\
lts__t_sectors.sum,lts__t_sectors.avg,lts__t_sectors.max,lts__t_sectors.min,\
dram__sectors_read.sum,dram__sectors_read.avg,dram__sectors_read.max,dram__sectors_read.min,\
dram__cycles_active.avg,dram__cycles_active.max"

# EIGHT separate processes: the effect is per-process, so one process is not a
# sample.  Six launches each, i.e. two full stages, so DIR0/1/2 are each seen
# twice and within-process stability can be checked as well.
for p in 1 2 3 4 5 6 7 8; do
  echo "=== PASS A proc $p"
  timeout 600 ncu --csv --clock-control none --metrics "$MA" \
    --kernel-name-base demangled --rename-kernels 0 \
    -k regex:tendency_p255_kernel -s 3 -c 6 \
    "$R/scale-dg_extraction" $CONF > $O/A_p$p.csv 2>$O/A_p$p.err
  echo "   exit=$? size=$(stat -c%s $O/A_p$p.csv 2>/dev/null)"
done
for p in 1 2 3 4; do
  echo "=== PASS B proc $p (metric-name validation + imbalance)"
  timeout 600 ncu --csv --clock-control none --metrics "$MB" \
    --kernel-name-base demangled --rename-kernels 0 \
    -k regex:tendency_p255_kernel -s 3 -c 6 \
    "$R/scale-dg_extraction" $CONF > $O/B_p$p.csv 2>$O/B_p$p.err
  rc=$?; echo "   exit=$rc size=$(stat -c%s $O/B_p$p.csv 2>/dev/null)"
  if [ $rc -ne 0 ]; then echo "   --- pass B stderr ---"; head -5 $O/B_p$p.err; break; fi
done
echo "ALL_DONE"
