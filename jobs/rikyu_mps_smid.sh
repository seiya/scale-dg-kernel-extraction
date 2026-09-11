#!/bin/bash
#SBATCH --job-name=dg_mps_smid
#SBATCH --gpus=1
#SBATCH --time=00:20:00
# Does CUDA_MPS_ACTIVE_THREAD_PERCENTAGE give the client that fraction of the
# SMs?  jobs/mps_smid_probe.cu counts the distinct %smid values that ran a
# block.  Same percentages as jobs/mps_balance.sh.
set -u
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
readonly OUT="mps_smid_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT/mps_pipe" "$OUT/mps_log"
module purge; module load nvhpc
nvcc -O3 -arch=sm_100 -o "$OUT/smid" jobs/mps_smid_probe.cu || exit 1
export CUDA_MPS_PIPE_DIRECTORY="$(cd "$OUT" && pwd)/mps_pipe"
export CUDA_MPS_LOG_DIRECTORY="$(cd "$OUT" && pwd)/mps_log"
stop() { echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; }
trap stop EXIT TERM INT
echo "--- without MPS (control)"
"$OUT/smid"
nvidia-cuda-mps-control -d || { echo "ERROR: no user MPS daemon here"; exit 1; }
sleep 2
for pct in 100 75 50 40 36 25; do
  printf "pct=%-4s " "$pct"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct "$OUT/smid"
done
stop
echo ALL_DONE
