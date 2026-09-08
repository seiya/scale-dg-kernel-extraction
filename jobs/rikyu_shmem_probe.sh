#!/bin/bash
#SBATCH -J dg_gb200_shmemprobe
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -o slurm-%j.out
#
# GB200 control for jobs/wisteria_shmem_probe.sh.  OUT OF ROLE (AGENTS.md):
# not a benchmark path, never reported as one.
#
# The control matters because the effect being chased does not exist here:
# measurement_inventory.md 21.15 measured the same p=511 FUSED_TC spread on
# GB200 at 0.08-0.22% in all three directions, against 2.3-2.9% on A100.  So
# the probe should be FLAT here.  If it is not, the probe is measuring its own
# noise and the A100 run cannot be read.
#
#   sbatch jobs/rikyu_shmem_probe.sh
set -u
module load nvhpc-hpcx
R="${SLURM_SUBMIT_DIR:-$PWD}"
O="$R/shmem_probe_gb200_${SLURM_JOB_ID:-manual}"
ARCH="${ARCH:-sm_100}"
PROCS="${PROCS:-12}"
REPS="${REPS:-3}"
mkdir -p "$O"; cd "$R" || exit 1
echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD) arch=$ARCH"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,clocks.max.sm --format=csv,noheader

PROBE="$R/jobs/shmem_latency_probe.cu"
if [ ! -f "$PROBE" ]; then
  echo "ERROR: $PROBE is missing on this machine." >&2
  echo "  Copy jobs/shmem_latency_probe.cu here and resubmit (TCS 9547915" >&2
  echo "  failed this way: nvcc reported only 'No such file or directory')." >&2
  exit 1
fi
nvcc -O3 -arch="$ARCH" -o "$O/shmem_probe" "$PROBE" \
  > "$O/build.log" 2>&1
echo "build rc=$?"
[ -x "$O/shmem_probe" ] || { sed 's/^/  | /' "$O/build.log"; exit 1; }

for p in $(seq 1 "$PROCS"); do
  echo "=== proc $p"
  timeout 300 "$O/shmem_probe" "$REPS" > "$O/probe_p$p.txt" 2>&1
  echo "   exit=$?"
  sed 's/^/   /' "$O/probe_p$p.txt"
done
echo "ALL_DONE  artifacts in $O"
