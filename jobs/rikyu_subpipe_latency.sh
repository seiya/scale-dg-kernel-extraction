#!/bin/bash
#SBATCH --job-name=dg_subpipe_lat
#SBATCH --gpus=1
#SBATCH --time=02:00:00
# The counters the latency term and the volume_flux L1 residual need.
#
# reports/subpipe_transfer_model.md 5F.1 shows the accounting closes:
#   W = (1 + sum stalls) * IPC        (Little's law, verified on job 92904)
# so a kernel that sits on no roof takes N_inst * (1 + sum stalls) / (4 W)
# SM-cycles.  That is an identity, not a prediction; the predictive content is
# whether (1 + sum stalls) transfers, and it does NOT -- volume_flux goes
# 106.7 -> 400.1 (A100) -> 308.2 (H100).  Decomposing it into
# sum_c (instruction fraction of class c) * (latency of class c) needs the
# long-latency population counted separately, which is the global loads that
# miss in L2.  Estimating it from sm__inst_executed_pipe_lsu instead gives
# 1.5-334 SM cycles within one machine, because most LSU instructions are
# shared-memory ops.
#
# Also adds the read/write-split global sector counts and l1tex__f_wavefronts,
# for the 13 volume_flux kernels whose l1tex__throughput the LSU wavefront roof
# misses by 13.2-13.9 points (section 5E.1).  sector/2 per SM-cycle reproduces
# them to 1.5% but overstates separable_lift by 29%, so the two are not on the
# same L1TEX stage and the split says which.
#
# Orders chosen for where the out-of-model family actually carries weight
# (section 5F, GB200): p=31-127 where eta>1.5 kernels are 18-94% of the call,
# plus p=255/511 where they are the elembnd_flux 0.2-3.2%.
set -u
export ORDERS="${ORDERS:-31 63 127 255 511}"
export PATHS="${PATHS:-CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM CUDAFORTRAN_FUSED}"
export SHAPES="${SHAPES:-8x8x4}"
export REPEATS="${REPEATS:-2}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export OUTDIR="${OUTDIR:-subpipe_latency_${SLURM_JOB_ID:-manual}}"
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
