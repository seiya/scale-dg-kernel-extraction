#!/bin/bash
#SBATCH --job-name=dg_gb200_dir1
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out
#
# measurement_inventory.md 21.13 narrowed the A100 p=511 variance to one
# launch: tendency_p255_kernel DIR1 moved 2.84% between two processes while
# DIR0 and DIR2 held to 0.2%.  That is the signature of per-process state, and
# physical page placement is the candidate.
#
# This is the GB200 CONTROL.  GB200's spread on this cell is 0.011%, so DIR1
# should NOT move here.  It also validates the metric names before the A100
# job asks for them: pass B uses .avg/.max/.min rollups that pass A does not,
# and a rejected name makes ncu fail outright, so they are in a separate
# invocation and pass A's data survives either way.
set -u
module load nvhpc-hpcx
export DEBUGINFOD_URLS=
R=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
O=$R/output/s37
mkdir -p "$O"; cd "$R" || exit 1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is) commit=$(git rev-parse --short HEAD)"

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
    $R/scale-dg_extraction $CONF > $O/A_p$p.csv 2>$O/A_p$p.err
  echo "   exit=$? size=$(stat -c%s $O/A_p$p.csv 2>/dev/null)"
done
for p in 1 2 3 4; do
  echo "=== PASS B proc $p (metric-name validation + imbalance)"
  timeout 600 ncu --csv --clock-control none --metrics "$MB" \
    --kernel-name-base demangled --rename-kernels 0 \
    -k regex:tendency_p255_kernel -s 3 -c 6 \
    $R/scale-dg_extraction $CONF > $O/B_p$p.csv 2>$O/B_p$p.err
  rc=$?; echo "   exit=$rc size=$(stat -c%s $O/B_p$p.csv 2>/dev/null)"
  if [ $rc -ne 0 ]; then echo "   --- pass B stderr ---"; head -5 $O/B_p$p.err; break; fi
done
echo "ALL_DONE"
