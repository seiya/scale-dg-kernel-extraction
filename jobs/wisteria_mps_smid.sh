#!/bin/bash
#PJM -N dg_a100_mps_smid
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=0:20:00
#PJM -g CHANGE_ME
#PJM -j
# Does CUDA_MPS_ACTIVE_THREAD_PERCENTAGE give this machine the fraction of SMs
# it is asked for?  On GB200 it does, to the whole SM (Slurm 96586): 152 / 114 /
# 76 / 60 / 54 / 38, so 40% is really 39.47% and 36% is 35.53%.  The A100 column
# of reports/subpipe_transfer_model.md 5Z assumes the same floor rounding and
# has not been checked; 5Z.4's matched-balance point depends on it.
set -u
ROOT="${PJM_O_WORKDIR:-$PWD}"; cd "$ROOT" || exit 1
OUT="a100_mps_smid_${PJM_JOBID:-manual}"; mkdir -p "$OUT/mps_pipe" "$OUT/mps_log"
module load nvidia/25.9
nvcc -O3 -arch=sm_80 -o "$OUT/smid" jobs/mps_smid_probe.cu || exit 1
export CUDA_MPS_PIPE_DIRECTORY="$(cd "$OUT" && pwd)/mps_pipe"
export CUDA_MPS_LOG_DIRECTORY="$(cd "$OUT" && pwd)/mps_log"
stop() { echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; }
trap stop EXIT TERM INT
echo "--- without MPS (control)"; "$OUT/smid"
nvidia-cuda-mps-control -d || { echo "ERROR: no user MPS daemon here"; exit 1; }
sleep 2
for pct in 100 75 50 41 40 39 36 25; do
  printf "pct=%-4s " "$pct"; CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct "$OUT/smid"
done
stop; echo ALL_DONE
