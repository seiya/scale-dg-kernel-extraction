#!/bin/bash
#PJM -N dg_a100_ctamap
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# measurement_inventory.md 21.19 leaves exactly one unknown on the A100 p=511
# FUSED_TC spread: WHY the waiting moves.  Section 21.16 and 21.18 established
# that eight sequential processes on ONE GPU split into a fast and a slow group
# that is fixed per process (2.2-2.3% between groups, 0.11% inside one), with
# an identical instruction count, identical DRAM bytes, identical occupancy --
# and 21.18 measured that short_scoreboard, mio_throttle and math_pipe_throttle
# rise while long_scoreboard falls.  The waiting moves from the global path to
# the shared/MIO one, with an access pattern that cannot differ.
#
# 21.19 names three candidates and marks all three unchecked:
#   (a) CTA-to-SM assignment order
#   (b) L1/shared carveout
#   (c) ECC / refresh
#
# WHAT ROUND 1 (TCS 9545646, node wa35, 8 processes) ALREADY SETTLED
# ------------------------------------------------------------------
# * (a) is DEAD.  sm__ctas_launched .sum and .avg are identical in all 24
#   launches (32 768 CTAs, 303.41 per SM) and .max/.min move 0.1-0.33% with no
#   consistent sign against duration (spearman +0.24 / -0.57 across the three
#   directions).  The CTAs land the same way in the fast and the slow group.
# * (c) is DEAD as far as the device reports it: the ECC counters and the row
#   remapper are zero and unchanged before the run and after every process.
# * Bank conflicts DO differ -- 9.0% in DIR1, 24.3% in DIR2, correlated with
#   duration (spearman +0.79 / +0.48) -- but they are FAR too few to matter.
#   The largest between-process difference is 2.6e4 conflicts against 2.7e8
#   shared wavefronts and 1.4e7 SM cycles: about 0.002% of the time, against a
#   2.3% effect.  A symptom, not the cause.
# * (b) came back UNTESTED because this script's own metric guard dropped every
#   launch__* name (they are launch statistics; --query-metrics does not list
#   them).  That is fixed below and is the first reason for round 2.
#
# WHAT ROUND 2 ADDS
# -----------------
# 1. The carveout, actually measured this time (launch__shared_mem_config_size).
# 2. The question round 1 sharpened: the SM burns 2.0-2.3% MORE CYCLES
#    (sm__cycles_active tracks duration at spearman +0.88 to +1.00), with an
#    instruction count identical to the digit and shared wavefronts identical
#    to 0.03%.  So the extra cycles are either work somewhere in the L1/MIO
#    path or waiting in front of it, and l1tex__cycles_active,
#    l1tex__lsu_writeback_active and the mio issue counters separate the two.
# 3. Twelve processes instead of eight, because the split is not a fixed
#    per-process property: in round 1 DIR1 had three FAST processes (5, 7, 8)
#    and DIR2 had three SLOW ones (2, 5, 4), i.e. proc 5 was on both sides.
#    The state is per (process, direction), which the earlier reading of a
#    per-process binary state did not anticipate.
#
# CONTROLS CARRIED FORWARD, SO THE GROUPS CAN BE IDENTIFIED AT ALL
# ---------------------------------------------------------------
# smsp__inst_executed.sum must stay identical to the digit (21.16); if it does
# not, this job is measuring something else and the rest is void.  The three
# stall ratios that moved are re-taken so that a process can be placed in the
# fast or the slow group from its own data, without assuming the split repeats
# the 4:4 of the previous job -- 21.16 already found the affected DIRECTION
# changing between jobs (DIR1 there, DIR2 in 21.18).
#
# PROTOCOL NOTES THAT ARE NOT OPTIONAL
# ------------------------------------
# * The FIRST launch ncu captures in a process is an outlier (21.16: 10 217 /
#   12 231 / 12 519 us against 9 78x for the second in the same process).  -s 3
#   -c 9 keeps three full stages so the first can be dropped in analysis, as
#   both earlier jobs did.
# * --clock-control none: the question is which resource binds in production,
#   and AGENTS.md forbids carrying a locked-clock percentage to boost.
# * timeout on every ncu invocation: a bad GPU node returns no output at all
#   rather than running slowly, so a long bound only delays the retry.  If a
#   process times out, check the node with pjstat and resubmit excluding it
#   before touching the profiler command.
#
# BEFORE SUBMITTING (cc80 / sm_80):
#   module unload nvidia && module load nvidia/25.9
#   make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
#   fill in -g above, then:  pjsub jobs/wisteria_p511_ctamap_ncu.sh
set -u
module unload nvidia >/dev/null 2>&1 || true
module load nvidia/25.9
export DEBUGINFOD_URLS=
R="${PJM_O_WORKDIR:-$PWD}"
O="$R/p511_ctamap_${PJM_JOBID:-manual}"
mkdir -p "$O"; cd "$R" || exit 1
echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,clocks.max.sm \
  --format=csv,noheader

# Freeze the executable: a concurrent make must not relink under the profiler.
cp -p "$R/scale-dg_extraction" "$O/exe" || exit 1
EXE="$O/exe"

CONF=$O/p511.conf
sed 's/DqdtKernel_Type *= *"[A-Z_]*"/DqdtKernel_Type = "CUDAFORTRAN_FUSED_TC"/' \
  "$R/namelists/perf_p511_gemm.conf" > "$CONF"

# Ask the installed ncu which metric names it has rather than assuming them:
# dfma_register_budget.md 21.5 records what a silently dropped metric costs.
timeout 300 ncu --query-metrics 2>/dev/null > "$O/allmetrics.txt"
have() { grep -qx "$1" "$O/allmetrics.txt" || grep -q "^$1\b" "$O/allmetrics.txt"; }
add()  { for m in "$@"; do
           base=${m%%.*}
           if have "$base"; then M="$M,$m"; else echo "MISSING METRIC: $m" >&2; fi
         done; }
# launch__* are LAUNCH STATISTICS, not hardware counters, and --query-metrics
# does not list them: job 9545646 asked for six of them and the guard above
# dropped all six, which is why the carveout candidate came back untested.
# They are passed through unfiltered; a name ncu does not know makes the whole
# invocation fail loudly, which is the failure mode this guard exists to avoid
# for counters and does not need to avoid for these.
add_launch() { for m in "$@"; do M="$M,$m"; done; }

M="gpu__time_duration.sum"
# control: must not move
add smsp__inst_executed.sum \
    sm__warps_active.avg.pct_of_peak_sustained_active
# (b) carveout and the launch geometry that would explain a different one
add_launch launch__shared_mem_config_size \
    launch__shared_mem_per_block_static \
    launch__shared_mem_per_block_dynamic \
    launch__registers_per_thread \
    launch__grid_size \
    launch__block_size \
    launch__occupancy_limit_shared_mem
# (a) CTA-to-SM assignment: imbalance in what each SM got, and the tail it makes
add sm__ctas_launched.sum sm__ctas_launched.avg sm__ctas_launched.max \
    sm__ctas_launched.min \
    sm__cycles_active.avg sm__cycles_active.max sm__cycles_active.min
# the mechanism 21.18 pointed at: shared-memory bank conflicts and the MIO queue
add l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum \
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
    l1tex__t_sector_hit_rate.pct \
    lts__t_sector_hit_rate.pct
# Is the extra time extra WORK in the L1/MIO path, or extra WAITING in front of
# it?  Job 9545646 established that the instruction stream and the shared
# wavefront counts are identical to 0.03% while the SM spends 2.2% more cycles,
# so the answer has to be one of these two and the counters separate them:
# l1tex__cycles_active rising with duration is work, staying flat is waiting.
add l1tex__cycles_active.sum \
    l1tex__lsu_writeback_active.sum \
    l1tex__throughput.avg.pct_of_peak_sustained_active \
    sm__mio_inst_issued.sum \
    sm__mioc_inst_issued.sum \
    sm__mio2rf_writeback_active.sum \
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    dram__bytes.sum
# the stalls that moved, so each process can be placed in a group from its own data
add smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio \
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio \
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio \
    smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
echo "metrics: $M" | tr ',' '\n' | sed 's/^/  /'

ecc_state() { # $1 label
  echo "--- ecc $1"
  nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total,retired_pages.pending \
    --format=csv,noheader 2>&1
  nvidia-smi -q -d ROW_REMAPPER 2>/dev/null | grep -E "Correctable|Uncorrectable|Pending|Remapping Failure" | sed 's/^/  /'
}

ecc_state "before-all" > "$O/ecc.txt"
# Twelve processes, not eight: job 9545646 found the split landing on DIR1
# (3 fast of 8) and DIR2 (3 slow of 8) with different members, so the shape
# of the distribution -- two groups or a continuum -- needs more draws.
for p in $(seq 1 "${PROCS:-12}"); do
  echo "=== proc $p"
  timeout 900 ncu --csv --clock-control none --metrics "$M" \
    --kernel-name-base demangled --rename-kernels 0 \
    -k regex:tendency_p255_kernel -s 3 -c 9 \
    "$EXE" "$CONF" > "$O/S_p$p.csv" 2>"$O/S_p$p.err"
  echo "   exit=$? size=$(stat -c%s "$O/S_p$p.csv" 2>/dev/null)"
  ecc_state "after-proc-$p" >> "$O/ecc.txt"
done
echo "ALL_DONE  artifacts in $O"
