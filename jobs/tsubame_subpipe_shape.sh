#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -N dg_h100_subpipe_shape
#$ -j y
#
# H100: what one DMMA instruction costs at each MMA shape.
#
# reports/subpipe_transfer_model.md 5A.1 reads the general law off the H100
# cuBLAS kernel: one DMMA instruction occupies the fp64+tensor pipe for
#     max( I_m , FLOP_instr / C_tc,m )     I_m = 4, C_tc = 256 on sm_90
# cycles, so DMMA.8x8x4 (512 FLOP) costs 4 cycles and reaches only 128
# FLOP/SM-cycle, while DMMA.16x8x8 (2048 FLOP) costs 8 and reaches 256.
# Measured there: 3.99-4.01 and 8.21-8.53.
#
# 16x8x4 is the shape the law says is the cheapest way to the full rate:
# max(4, 1024/256) = 4 cycles for 1024 FLOP, i.e. 256 FLOP/SM-cycle at the same
# instruction cost as 8x8x4.  Nothing in the tree has measured that cell, and
# it is an INDEPENDENT test of the law -- the 16x8x8 branch was read off the
# library kernel, this one is a prediction.
#
# Only the CUTLASS paths take CutlassMmaShape, and only p=511 is needed, so
# this is four ncu invocations (2 shapes x 2 paths x 1 order), a few minutes.
#
#   qsub -g <GROUP> jobs/tsubame_subpipe_shape.sh
#
# REQUIRES the current jobs/subpipe_ncu.sh, the one with the SHAPES knob.  The
# copy that ran SGE 8621140 predates it and ignores SHAPES silently.
set -u
export ORDERS="${ORDERS:-511}"
export PATHS="${PATHS:-CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM_CUTE}"
export SHAPES="${SHAPES:-8x8x4 16x8x4}"
export REPEATS="${REPEATS:-2}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-2700}"
export OUTDIR="${OUTDIR:-h100_subpipe_shape_${JOB_ID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
