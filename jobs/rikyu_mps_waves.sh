#!/bin/bash
#SBATCH --job-name=dg_mps_waves
#SBATCH --gpus=1
#SBATCH --time=01:00:00
# The control reports/subpipe_transfer_model.md 5Z.5 says is missing: throttling
# the SMs also raises waves per SM, so the balance effect and the wave effect
# are confounded in 5Z.
#
# Ne separates them.  At p=63 the element count per dimension is 4; running
# Ne = 1, 2 and 4 changes the grid by 64x and 8x -- hence waves per SM -- while
# the machine balance (DRAM bytes per SM-clock) does not move at all.  Crossing
# that with the MPS percentage gives a 2-D grid in (waves, balance), and the
# question "is 5Z.3 a wave artefact" becomes a coefficient rather than a worry.
#
# Ne < 4 shrinks the problem, so the absolute times are not comparable to the
# published p=63 cells; only the ratios within one (Ne, pct) cell are.
set -u
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
readonly OUT="mps_waves_${SLURM_JOB_ID:-manual}"
readonly NES="${NES:-4 2 1}"
readonly PCTS="${PCTS:-100 50 25}"
readonly PATHS="${PATHS:-CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM}"
readonly ROUNDS="${ROUNDS:-3}"
mkdir -p "$OUT/config" "$OUT/logs" "$OUT/mps_pipe" "$OUT/mps_log"
module purge; module load nvhpc
[ -x ./scale-dg_extraction ] || { echo "ERROR: build first" >&2; exit 1; }
cp -p ./scale-dg_extraction "$OUT/scale-dg_extraction.frozen"
EXE="$(cd "$OUT" && pwd)/scale-dg_extraction.frozen"
export CUDA_MPS_PIPE_DIRECTORY="$(cd "$OUT" && pwd)/mps_pipe"
export CUDA_MPS_LOG_DIRECTORY="$(cd "$OUT" && pwd)/mps_log"
stop() { echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; }
trap stop EXIT TERM INT
nvidia-cuda-mps-control -d || { echo "ERROR: no user MPS daemon here" >&2; exit 1; }
sleep 2
{ date --iso-8601=seconds; echo "commit=$(git rev-parse HEAD)"; echo "nes=$NES pcts=$PCTS rounds=$ROUNDS"; } > "$OUT/metadata.txt"
for ne in $NES; do
  for path in $PATHS; do
    c="$OUT/config/ne${ne}_${path#CUDAFORTRAN_}.conf"
    sed -e "s/DqdtKernel_Type *= *\"[A-Z0-9_]*\"/DqdtKernel_Type = \"$path\"/" \
        -e "s/NeX *= *[0-9]*/NeX = $ne/" -e "s/NeY *= *[0-9]*/NeY = $ne/" \
        -e "s/NeZ *= *[0-9]*/NeZ = $ne/" namelists/perf_p63_gemm.conf > "$c"
  done
done
for r in $(seq 1 "$ROUNDS"); do
  for ne in $NES; do for pct in $PCTS; do for path in $PATHS; do
    tag="r${r}_ne${ne}_${pct}pct_${path#CUDAFORTRAN_}"
    out=$(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct timeout 600 "$EXE" \
          "$OUT/config/ne${ne}_${path#CUDAFORTRAN_}.conf" 2>&1); rc=$?
    printf '%s\n' "$out" > "$OUT/logs/${tag}.log"
    printf "%-40s rc=%-3s stage=%s\n" "$tag" "$rc" \
      "$(printf '%s\n' "$out" | awk '/Step loop per stage:/{print $NF}')"
  done; done; done
done
stop; echo "ALL_DONE artifacts in $OUT"
