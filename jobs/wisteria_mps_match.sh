#!/bin/bash
#PJM -N dg_a100_mps_match
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=0:30:00
#PJM -g CHANGE_ME
#PJM -j
# The one point reports/subpipe_transfer_model.md 5Z.7 needs: A100 at 41%,
# which is 44 SMs = 25.06 DRAM bytes per SM-clock against GB200's 25.21 (-0.6%).
# The balance sweep ran 40%, which the smid probe showed is 42 SMs = 26.25, i.e.
# 4.1% leaner, so the matched-balance comparison has never been taken at the
# matched balance.  p=511, the two paths of the main ratio, three rounds.
set -u
ROOT="${PJM_O_WORKDIR:-$PWD}"; cd "$ROOT" || exit 1
OUT="a100_mps_match_${PJM_JOBID:-manual}"; mkdir -p "$OUT/config" "$OUT/logs" "$OUT/mps_pipe" "$OUT/mps_log"
module load nvidia/25.9
[ -x ./scale-dg_extraction ] || { echo "ERROR: build first" >&2; exit 1; }
cp -p ./scale-dg_extraction "$OUT/scale-dg_extraction.frozen"
EXE="$(cd "$OUT" && pwd)/scale-dg_extraction.frozen"
export CUDA_MPS_PIPE_DIRECTORY="$(cd "$OUT" && pwd)/mps_pipe"
export CUDA_MPS_LOG_DIRECTORY="$(cd "$OUT" && pwd)/mps_log"
stop() { echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; }
trap stop EXIT TERM INT
nvidia-cuda-mps-control -d || { echo "ERROR: no user MPS daemon here" >&2; exit 1; }
sleep 2
for path in CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED; do
  sed -e "s/DqdtKernel_Type *= *\"[A-Z0-9_]*\"/DqdtKernel_Type = \"$path\"/" \
      namelists/perf_p511_gemm.conf > "$OUT/config/${path#CUDAFORTRAN_}.conf"
done
for r in 1 2 3; do
  for pct in 41 40; do
    for path in CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED; do
      tag="r${r}_${pct}pct_${path#CUDAFORTRAN_}"
      out=$(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct timeout 900 "$EXE" \
            "$OUT/config/${path#CUDAFORTRAN_}.conf" 2>&1); rc=$?
      printf '%s\n' "$out" > "$OUT/logs/${tag}.log"
      printf "%-36s rc=%-3s stage=%s\n" "$tag" "$rc" \
        "$(printf '%s\n' "$out" | awk '/Step loop per stage:/{print $NF}')"
    done
  done
done
stop; echo ALL_DONE
