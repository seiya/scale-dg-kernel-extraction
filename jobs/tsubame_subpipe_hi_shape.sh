#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:30:00
#$ -N dg_h100_subpipe_hi_shape
#$ -j y
#
# H100 sub-pipe sweep at the PRODUCTION MMA shape (16x8x4) for the high
# orders, which is the hole reports/subpipe_transfer_model.md 5AD.6 records.
#
# What is missing and why it matters.  H100's production CutlassMmaShape is
# 16x8x4 (section 5X), and the 16x8x4 sweep in the tree stops at p=575
# (SGE 8621936, GEMM_FUSED only; plus p=511 GEMM_CUTE from SGE 8621412).  At
# p=767 only 8x8x4 exists (SGE 8630131 / 8631255).  So the p=767 row of
# section 5AD.1e can state GEMM_FUSED@16x8x4 against production (-24.8%) but
# CANNOT score any RATIO at the production shape: the denominators
# (FUSED_TC, GEMM, GEMM_CUTE) have no 16x8x4-tagged measurement, and
# analysis/subpipe_predict.py looks the target up by
#     p<p>_<PATH>_<shape>_none_r*_{a,b}.csv
# so a missing tag is a skipped cell, not a fallback.
#
# Note that CutlassMmaShape is IGNORED unless the path is GEMM_FUSED or
# GEMM_CUTE (mod_advect3d_eq.f90, which prints a line saying so).  Taking
# FUSED_TC and GEMM under SHAPES=16x8x4 is therefore two things at once:
#   * it supplies the tagged files the ratio scorer needs, and
#   * it is a CONTROL -- those two rows must reproduce the 8x8x4 rows of
#     SGE 8630131 kernel by kernel.  If they do not, the difference is the
#     machine or the tree, not the shape, and every 16x8x4 conclusion at this
#     order has to wait for that to be explained.
# That is also why all four paths are taken in one job rather than two: an
# interleaved pair on one GPU is the only comparison AGENTS.md accepts.
#
# Stage 2 (p=1023) is the other half of the 5AD.6 bullet and is expected to
# FAIL for capacity, not for time: p=1023 is Np = 2^30 nodes, and
# reports/p1023_gap_study.md section 1 records that it needed volume-sized
# temporaries removed to fit GB200's 189471 MiB.  This H100 has 95830 MiB.
# It runs last, after stage 1 has already been written out, so an OOM costs
# nothing and settles whether the p=1023 half of the hole is closable on this
# machine at all.  Set HI_P1023=0 to skip it.  p=1023 has no FUSED_TC
# (AGENTS.md: p=1023 is GEMM-only), so its path list is the GEMM family.
#
#   qsub -g <GROUP> jobs/tsubame_subpipe_hi_shape.sh
#
# REQUIRES the current jobs/subpipe_ncu.sh (SHAPES knob, L1 / latency
# counters, flux_all in KERNELS) and a CUDA build in the checkout.
# OUTDIR matches tsubame/h100_subpipe_orders_* so analysis/subpipe_predict.py
# picks the directory up by dropping it in place, with no edit to SWEEPS.
set -u
ROOT="${SCALE_DG_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1

export SHAPES="${SHAPES:-16x8x4}"
export REPEATS="${REPEATS:-1}"
export CLOCKS="${CLOCKS:-none}"
export NSTEP="${NSTEP:-4}"
readonly TAG="${JOB_ID:-manual}"

# Stage 1: p=767, the four paths that the section 5AD.1e ratios need.
ORDERS=767 \
PATHS="CUDAFORTRAN_FUSED_TC CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE" \
TIME_BUDGET="${TIME_BUDGET_P767:-3000}" \
OUTDIR="h100_subpipe_orders_16x8x4_p767_${TAG}" \
  bash jobs/subpipe_ncu.sh
echo "STAGE1_RC=$?"

if [ "${HI_P1023:-1}" = 1 ]; then
  echo
  echo "===== stage 2: p=1023 (expected to hit the 95830 MiB limit) ====="
  ORDERS=1023 \
  PATHS="CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE" \
  TIME_BUDGET="${TIME_BUDGET_P1023:-1800}" \
  OUTDIR="h100_subpipe_orders_16x8x4_p1023_${TAG}" \
    bash jobs/subpipe_ncu.sh
  echo "STAGE2_RC=$?"
fi
