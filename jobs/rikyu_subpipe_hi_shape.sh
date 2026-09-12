#!/bin/bash
#SBATCH --job-name=dg_subpipe_hi_shape
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#
# GB200 at p=767, both CutlassMmaShapes, GEMM_CUTE and GEMM_FUSED.
#
# Two things at once, and they need the same cells:
#
# 1. THE CALIBRATION GAP.  analysis/subpipe_inputs.csv has no
#    (GB200, 767, GEMM_CUTE) row -- GB200's high-order sweep
#    (jobs/rikyu_subpipe_hi.sh, Slurm 94874) ran FUSED_TC / GEMM_FUSED / GEMM
#    only.  A100 cannot hold p=767 at all (40 GB), so GB200 is the ONLY
#    calibration source there.  That is why
#    reports/subpipe_transfer_model.md 5AD.7c has to print "predictable: no"
#    for GEMM_CUTE@767 even though SGE 8648736 measured the H100 side
#    (67903.4 us).  One 8x8x4 cell here makes that a scored cell.
#
# 2. THE SHAPE CONTROL AT A HIGH ORDER.  On sm_100 a requested 16x8x4 lowers
#    to two DMMA.8x8x4 (reports/sm90_mma_shape_survey.md 6), so the knob must
#    do nothing on GB200.  Slurm 92628 measured exactly that at p=511:
#    tensor instructions identical (268435456 at both shapes) and total time
#    +0.24% (GEMM_FUSED) / +0.04% (GEMM_CUTE).  p=767 is where the H100 side
#    of the same knob is largest -- -31.0% in cycles (section 5AD.7b) -- so it
#    is the order at which "the calibration machine has no shape knob"
#    (section 5AD.6, third bullet) is worth stating with a measurement rather
#    than by extrapolating the p=511 control.
#
# Both shapes of both paths run in ONE job so the comparison is an
# interleaved A/B on a GPU the job owns, as AGENTS.md requires; the existing
# GB200 8x8x4 rows at this order come from a different job and node.
#
#   sbatch jobs/rikyu_subpipe_hi_shape.sh
#
# p=1023 is deliberately absent: H100 cannot hold it (SGE 8648736 OOMed at
# 95830 MiB, section 5AD.7d), so a GB200 calibration cell there has no
# transfer target.
set -u
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
export ORDERS="${ORDERS:-767}"
export PATHS="${PATHS:-CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED}"
export SHAPES="${SHAPES:-8x8x4 16x8x4}"
export REPEATS="${REPEATS:-1}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export NCU_TIMEOUT="${NCU_TIMEOUT:-1500}"
export OUTDIR="${OUTDIR:-subpipe_hi_shape_${SLURM_JOB_ID:-manual}}"
exec bash jobs/subpipe_ncu.sh
