#!/bin/bash
#SBATCH --job-name=dg_p511_rev_ncu
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#
# Why does the p=511 ranking invert between the machines?
#
#   GB200   GEMM_FUSED 38.42 ms/step beats FUSED_TC 40.43   (ratio 0.950)
#   A100    FUSED_TC   96.71        beats GEMM_FUSED 102.98 (ratio 1.065)
#
# Both paths issue FP64 tensor instructions, so the Tensor roof is not what
# separates them.  The standing hypothesis is traffic: reports/README.md puts
# GEMM_FUSED at 145.3 B/node against 64.5 unique, while the TC fused paths are
# near 66 at p=255, so GEMM_FUSED moves roughly 2.2x the DRAM bytes.  A100 has
# 10.2 bytes of HBM per SM-cycle against GB200's 25.2 (a100_report.md 13.3), so
# the same extra traffic should cost it about 2.5x more.
#
#   H1 predicts: dram__bytes.sum for GEMM_FUSED / FUSED_TC is about 2.2 on both
#   machines, and A100's GEMM_FUSED dram% is roughly twice its FUSED_TC dram%.
#   If the byte ratio comes out near 1, H1 is dead and the cause is elsewhere
#   (occupancy, register pressure, the epilogue's L2 behaviour).
#
# ncu locks clocks, so this identifies the mechanism only; the ranking itself
# is already a wall-time result from the interleaved path sweeps (AGENTS.md).
#
# KERNELS is widened because the default regex matches only the hand-written
# fused kernels; without cutlass/cublas in it the GEMM_FUSED path would show no
# volume GEMM at all.  LAUNCH_COUNT is cut to 4 and REPEATS to 2 because a
# p=511 kernel runs for tens of milliseconds and ncu replays it per pass.
#
#   sbatch jobs/rikyu_p511_reversal_ncu.sh
set -u
export ORDERS="${ORDERS:-511}"
export PATHS="${PATHS:-CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM}"
export KERNELS="${KERNELS:-regex:tendency_|pdl_elembnd|cutlass|gemm|Kernel}"
export LAUNCH_SKIP="${LAUNCH_SKIP:-4}"
export LAUNCH_COUNT="${LAUNCH_COUNT:-4}"
export REPEATS="${REPEATS:-2}"
export CLOCKS="${CLOCKS:-none}"
export NSTEP="${NSTEP:-4}"
export NCU_TIMEOUT="${NCU_TIMEOUT:-900}"
export TIME_BUDGET="${TIME_BUDGET:-4800}"
export OUTDIR="${OUTDIR:-p511_reversal_ncu_${SLURM_JOB_ID:-manual}}"
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/l1_roof_ncu.sh
