#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dg_h100_subpipe_cute_shape
#$ -j y
#
# H100 GEMM_CUTE at the production MMA shape, every order below 767.
#
# reports/subpipe_transfer_model.md 5AD.6, second bullet: at 16x8x4 the tree
# has GEMM_CUTE at p=511 only (SGE 8621412), now joined by p=767 (SGE
# 8648736).  Everything in between is GEMM_FUSED-only, because the 16x8x4
# order sweep (SGE 8621936) ran that path alone.  So the FUSION axis --
# GEMM_CUTE vs GEMM_FUSED, the one comparison the two paths exist to make --
# can only be taken at the production shape at two isolated orders, and the
# 575 row of section 5AD.1e has to fall back on production wall time.
#
# This job takes the missing orders.  p=511 and p=767 are left out: they are
# measured, and re-taking them here would put them on a different node from
# the GEMM_FUSED rows they are compared against, which is the systematic that
# section 5AD.7a quantified (cuBLAS/CUTLASS kernels clock down by a
# node-dependent 4%; only the FP64-tensor-heavy kernels do).
#
#   qsub -g <GROUP> jobs/tsubame_subpipe_cute_shape.sh
#
# The comparison partner for every cell is
# tsubame/h100_subpipe_orders_16x8x4/ (SGE 8621936, GEMM_FUSED at the same
# orders and shape), so read the pair as a cross-job ratio with that 4% in
# mind.  REQUIRES the current jobs/subpipe_ncu.sh and a CUDA build.
set -u
ROOT="${SCALE_DG_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
export ORDERS="${ORDERS:-7 15 31 63 127 255 575}"
export PATHS="${PATHS:-CUDAFORTRAN_GEMM_CUTE}"
export SHAPES="${SHAPES:-16x8x4}"
export REPEATS="${REPEATS:-1}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export OUTDIR="${OUTDIR:-h100_subpipe_orders_16x8x4_cute_${JOB_ID:-manual}}"
exec bash jobs/subpipe_ncu.sh
