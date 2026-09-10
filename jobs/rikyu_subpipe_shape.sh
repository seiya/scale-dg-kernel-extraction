#!/bin/bash
#SBATCH --job-name=dg_subpipe_shape
#SBATCH --gpus=1
#SBATCH --time=01:00:00
# GB200 control for the shared-pipe pricing law.
#
# reports/sm90_mma_shape_survey.md 6 counts the SASS: on sm_100 a PTX
# mma.sync.m16n8k4.f64 lowers to TWO DMMA.8x8x4, so switching CutlassMmaShape
# from 8x8x4 to 16x8x4 must leave the DMMA instruction count -- and therefore
# the shared-pipe cycles, and therefore the time -- unchanged on GB200, while
# on H100 the same switch issues one DMMA.16x8x4 and halves them.  This job
# measures the GB200 half of that statement with the sub-pipe counters, which
# is a calibration-machine test of the pricing law rather than a timing fit.
#
# GEMM_CUTE is included because tsubame/roof_bytes_h100_libgap_8x8x4 (SGE
# 8611482) holds the H100 side of that path at 8x8x4, which is the only
# iso-shape H100 comparison available for a CUTLASS mainloop.
set -u
export ORDERS="${ORDERS:-511}"
export PATHS="${PATHS:-CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_GEMM_CUTE}"
export SHAPES="${SHAPES:-8x8x4 16x8x4}"
export REPEATS="${REPEATS:-2}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-2700}"
export OUTDIR="${OUTDIR:-subpipe_shape_${SLURM_JOB_ID:-manual}}"
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
