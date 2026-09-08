#!/bin/bash
#PJM -N dg_a100_shmemprobe
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# OUT-OF-ROLE ABLATION (AGENTS.md).  Nothing here is a benchmark path, and no
# number from this job may be reported as a path result.
#
# measurement_inventory.md 23.6 ends with one question that no ncu counter can
# answer: twelve sequential processes of the SAME p=511 FUSED_TC binary on the
# SAME A100 differ 2.3-2.9%, with every counted quantity identical and the
# extra time appearing only as short_scoreboard / mio_throttle waiting.  Is
# that varying shared-memory latency a property of THIS KERNEL or of the
# MACHINE?
#
# jobs/shmem_latency_probe.cu removes the kernel: a dependent chain of
# shared-memory loads, no DG arithmetic, no mma, launched with the same grid
# (32 768), block (128) and static shared (32 768 B) that section 23.1
# measured, so CTA scheduling and the carveout are not free variables.  A
# global-memory chain runs beside it as the control on the other path -- 23.2
# found long_scoreboard FALLING while short_scoreboard rose, so the global side
# is expected to stay flat.
#
# Twelve processes, run exactly as sections 22 and 23 ran them: sequentially,
# in one job, on one GPU.
#
# Predictions frozen in output/s44/PREDICTIONS.txt before submission.
#
# BEFORE SUBMITTING:
#   module unload nvidia && module load nvidia/25.9
#   fill in -g above, then:  pjsub jobs/wisteria_shmem_probe.sh
# (no repository build is needed; the job compiles the probe itself)
set -u
module unload nvidia >/dev/null 2>&1 || true
module load nvidia/25.9
R="${PJM_O_WORKDIR:-$PWD}"
O="$R/shmem_probe_${PJM_JOBID:-manual}"
ARCH="${ARCH:-sm_80}"
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
