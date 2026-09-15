#!/bin/bash
# Machine balance as a knob, on one silicon.  Machine-independent body; run it
# through a wrapper:
#
#   RIKYU:    sbatch jobs/rikyu_mps_balance.sh
#   TSUBAME:  qsub -g <GROUP> jobs/tsubame_mps_balance.sh
#   Wisteria: pjsub jobs/wisteria_mps_balance.sh
#
# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE limits the SMs a client may use while HBM
# stays whole, so DRAM bytes per SM-clock -- the coordinate that decides every
# crossover in reports/subpipe_transfer_model.md -- scales as 1/x with NOTHING
# else moving: same ISA, same compiler, same library kernel selection, same
# clock, same occupancy per SM.  The clock experiment of section 5P could not
# do this: dropping the clock moves the unloaded latency in cycles by the same
# factor, so the two coordinates are inseparable there (5P.1).  SM count moves
# only one of them.
#
#                   100%    75%    50%    40%    36%    25%
#   GB200 25.21 ->  25.2   33.6   50.4   63.0   70.0  100.8   B/SM-clk
#   A100  10.21 ->  10.2   13.6   20.4   25.5   28.4   40.8
#   H100   9.17 ->   9.2   12.2   18.3   22.9   25.5   36.7
#
# The percentages are chosen so the machines MEET: A100 at 40% and H100 at 36%
# both sit at GB200's 25.2 B/SM-clk.  A100 at 40% additionally has the same
# fp64 tensor throughput per SM-clock as GB200 (128.1 against 127.9), so those
# two points differ in the CUDA-core coordinate and almost nothing else -- the
# degeneracy of section 5M.4, undone by construction rather than by a fourth
# machine.
#
# No profiler: the production timers are the yardstick (the same "CUDA device"
# sum, "Step loop per stage" and "Main per step" jobs/tsubame_paths.sh reports),
# which also avoids profiling MPS clients.
#
# Whether a user may start an MPS daemon is a site policy and differs per
# machine, so this fails fast and says so rather than falling back to a
# measurement that silently has no SM limit.
set -u

# Freeze: editing a running job script shifts bash's byte offsets and kills it.
if [ -z "${SCALE_DG_FROZEN_SCRIPT:-}" ] && [ -r "$0" ]; then
  frozen_self=$(mktemp "${TMPDIR:-/tmp}/mps_balance.XXXXXX") || exit 1
  cat "$0" > "$frozen_self" || exit 1
  export SCALE_DG_FROZEN_SCRIPT="$frozen_self"
  bash "$frozen_self" "$@"; rc=$?; rm -f "$frozen_self"; exit $rc
fi

ROOT=""
for c in "${SCALE_DG_ROOT:-}" "${SGE_O_WORKDIR:-}" "${PJM_O_WORKDIR:-}" "$PWD"; do
  [ -n "$c" ] && [ -f "$c/namelists/perf_p63_gemm.conf" ] && { ROOT="$c"; break; }
done
[ -n "$ROOT" ] || { echo "ERROR: checkout not found" >&2; exit 1; }
export SCALE_DG_ROOT="$ROOT"
cd "$ROOT" || exit 1
readonly JOB_TAG=${SLURM_JOB_ID:-${JOB_ID:-${PJM_JOBID:-manual}}}
readonly OUTDIR=${OUTDIR:-"mps_balance_${JOB_TAG}"}
readonly ROUNDS_DEFAULT=${ROUNDS:-3}
readonly ORDERS=$(printf '%s' "${ORDERS:-63 255 511}" | tr ':,' '  ')
readonly PATHS=$(printf '%s' "${PATHS:-CUDAFORTRAN_FUSED_TC CUDAFORTRAN_FUSED CUDAFORTRAN_FUSED_DFMA CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED}" | tr ':,' '  ')
# 40% and 36% are not arbitrary: they put A100 and H100 at GB200's 25.21
# B/SM-clk, so the same percentages run on the other two machines give matched
# balance points across silicon.  Here they are simply more of the sweep.
readonly PCTS=$(printf '%s' "${PCTS:-100 75 50 40 36 25}" | tr ':,' '  ')
readonly ROUNDS=${ROUNDS:-3}
readonly TIME_BUDGET=${TIME_BUDGET:-9000}
readonly T0=$(date +%s)

[ "${SCALE_DG_MODULE_PURGE:-1}" = 1 ] && module purge
for m in ${SCALE_DG_MODULES:-nvhpc}; do module load "$m"; done
mkdir -p "$OUTDIR/config" "$OUTDIR/logs" "$OUTDIR/mps_pipe" "$OUTDIR/mps_log" || exit 1
[ -x ./scale-dg_extraction ] || { echo "ERROR: build first" >&2; exit 1; }
cp -p ./scale-dg_extraction "$OUTDIR/scale-dg_extraction.frozen"
readonly EXE="$(cd "$OUTDIR" && pwd)/scale-dg_extraction.frozen"

export CUDA_MPS_PIPE_DIRECTORY="$(cd "$OUTDIR" && pwd)/mps_pipe"
export CUDA_MPS_LOG_DIRECTORY="$(cd "$OUTDIR" && pwd)/mps_log"
stop_mps() { echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; }
trap stop_mps EXIT TERM INT
if ! nvidia-cuda-mps-control -d; then
  echo "ERROR: this site does not let a user start an MPS daemon." >&2
  echo "       Without it CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is ignored and the" >&2
  echo "       sweep would measure the same machine six times." >&2
  exit 1
fi
sleep 2
echo get_server_list | nvidia-cuda-mps-control

# Record the SM count each percentage actually grants, in this job, on this
# node.  The percentage-to-SM mapping is a machine property with TPC (2 SM)
# granularity that differs from the obvious floor on A100 (section 5Z.7), and
# borrowing it from another job is the cross-job assumption that left the
# "matched balance" point of 5Z.3 4.1% off the balance it claimed to match.
if nvcc -O3 -arch=native -o "$OUTDIR/smid" jobs/mps_smid_probe.cu 2>/dev/null \
   || nvcc -O3 -o "$OUTDIR/smid" jobs/mps_smid_probe.cu; then
  for pct in $PCTS; do
    printf "smid pct=%-4s " "$pct"
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct "$OUTDIR/smid"
  done | tee "$OUTDIR/smid.txt"
else
  echo "WARNING: smid probe did not build; SM counts must be taken from another job" | tee "$OUTDIR/smid.txt"
fi

{
  echo "date=$(date --iso-8601=seconds) job=$JOB_TAG host=$(hostname)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "orders=$ORDERS rounds=$ROUNDS pcts=$PCTS"
  echo "paths=$PATHS"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.max.sm --format=csv,noheader
} > "$OUTDIR/metadata.txt"
cat "$OUTDIR/metadata.txt"

for p in $ORDERS; do
  for path in $PATHS; do
    conf="$OUTDIR/config/p${p}_${path#CUDAFORTRAN_}.conf"
    sed -e "s/DqdtKernel_Type *= *\"[A-Z0-9_]*\"/DqdtKernel_Type = \"$path\"/" \
        "namelists/perf_p${p}_gemm.conf" > "$conf"
  done
done

# Warm up before the first measured run.  The first run of a job follows a
# 10-12 s cold start (shared libraries, cuBLAS init), and on A100 the runs in
# the first 10-15 s after that are slower by a fixed 5-20 ms each -- 3-15% of
# a p=63 run (reports/subpipe_transfer_model.md 5AG.3.2c / 5AG.3.5): with a
# single discarded run, TCS 9631289 still had its whole first p=63 round slow.
# Three rounds' median discards one such round; a two-round sweep cannot,
# which is how 9580356's 100% p=63 FUSED_TC / FUSED came out 2.7 / 5.7% high.
# So run the first configuration, discarded, for at least WARMUP_S seconds.
first_p=${ORDERS%% *}; first_path=${PATHS%% *}
w0=$(date +%s); n=0
while :; do
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${PCTS%% *} timeout 900 "$EXE" \
    "$OUTDIR/config/p${first_p}_${first_path#CUDAFORTRAN_}.conf" > "$OUTDIR/logs/warmup.log" 2>&1
  n=$((n + 1))
  [ $(( $(date +%s) - w0 )) -ge "${WARMUP_S:-30}" ] && break
done
echo "warmup: $n runs of p${first_p} ${first_path#CUDAFORTRAN_} over $(( $(date +%s) - w0 )) s (discarded)"

# Rounds outermost and percentage inside, so a drift in the machine hits every
# percentage equally -- the same interleaving rule the path sweeps use.
# clk= is the SM clock nvidia-smi reports just before the run starts: the
# direct check on whether a slow run started at the idle clock (5AG.3.2c).
for r in $(seq 1 "$ROUNDS"); do
  for p in $ORDERS; do
    for pct in $PCTS; do
      for path in $PATHS; do
        now=$(date +%s); [ $((now - T0)) -lt "$TIME_BUDGET" ] || { echo "TIME_BUDGET reached"; break 4; }
        tag="r${r}_p${p}_${pct}pct_${path#CUDAFORTRAN_}"
        conf="$OUTDIR/config/p${p}_${path#CUDAFORTRAN_}.conf"
        # SM clock, memory clock, active throttle reasons, temperature, power
        # and the host's 1-minute load average just before the run.  The SM
        # clock alone did not explain the 2026-09-15 evening A100 jobs
        # (9624982 / 9624991 / 9625160), which ran p<=255 4-16% slower than
        # every earlier job on two nodes at once while p=511 and the SM clock
        # were unchanged.  UseCudaGraph is off in these namelists, so a host
        # starved of CPU by a co-tenant on a shared node shows up as launch
        # gaps inside the device-timed window, and short-kernel orders suffer
        # first -- the load average is the cheap witness for that.
        # clocks_throttle_reasons.* was renamed clocks_event_reasons.* in
        # driver 535 (the old name is an alias again in newer drivers), so
        # try both.
        smi="nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i ${CUDA_VISIBLE_DEVICES%%,*}}"
        clk=$($smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw \
                --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        thr=$($smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>/dev/null | head -1)
        case "$thr" in 0x*) ;; *) thr=$($smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1);; esac
        case "$thr" in 0x*) ;; *) thr=-;; esac
        clk="${clk:--},${thr},load=$(cut -d' ' -f1 /proc/loadavg)"
        out=$(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct timeout 900 "$EXE" "$conf" 2>&1)
        rc=$?
        printf '%s\n' "$out" > "$OUTDIR/logs/${tag}.log"
        printf "%-40s rc=%-3s clk=%-40s dev=%-12s stage=%-12s main=%s\n" "$tag" "$rc" "${clk:--}" \
          "$(printf '%s\n' "$out" | awk '/CUDA device/{s+=$NF} END{if (s>0) printf "%.5E", s; else printf "-"}')" \
          "$(printf '%s\n' "$out" | awk '/Step loop per stage:/{print $NF}')" \
          "$(printf '%s\n' "$out" | awk '/Main per step:/{print $NF}')"
      done
    done
  done
done
stop_mps
echo "ALL_DONE artifacts in $OUTDIR"
