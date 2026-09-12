#!/bin/bash
#SBATCH --job-name=dg_subpipe_rep
#SBATCH --gpus=1
#SBATCH --time=02:00:00
# Run-to-run spread of the sweep, which reports/subpipe_transfer_model.md 5W.4
# needs: every sweep so far is REPEATS=1, and half the cells the model "misses"
# have true gaps under 2%.  Whether those are model misses or measurement noise
# is not decidable without this.
#
# TWO SETS, same body; the wrappers of the other two machines
# (jobs/wisteria_subpipe_repeat.sh, jobs/tsubame_subpipe_repeat.sh) take the
# same knobs, because 5Y.2 puts the three machines' widths in one table.
#
#   (1) HIGH ORDER, the default below -- the S/N < 1 cells of 5W.4 sit on the
#       library axis and on the p=511/575 dead heats.  Measured: Slurm 95805
#       (GB200), TCS 9579033 (A100), SGE 8631255 (H100, 2 rounds only).
#
#         sbatch jobs/rikyu_subpipe_repeat.sh
#
#   (2) LOW ORDER, 5Y.5's remaining hole -- the width was never measured at
#       p = 7/15/31 on any machine, and four of the sixteen misses that hold
#       the a priori S/N >= 1.1 rule to 94.0% are low-order cells whose true
#       gap is 0.98-2.58% (GB200 p=7 GEMM_CUTE/GEMM_FUSED 0.98%, H100 p=15
#       FUSED_DFMA/FUSED 1.67%, A100 p=7 GEMM_CUTE/GEMM 2.17%, GB200 p=31
#       GEMM_CUTE/GEMM 2.58%).  At the high-order width (ratio spread <= 1.3%)
#       the first is a tie, not a miss; at 3% all four are.  All six
#       transferred paths are needed because those four cells use five of
#       them:
#
#         ORDERS="7 15 31" \
#         PATHS="CUDAFORTRAN_FUSED CUDAFORTRAN_FUSED_TC CUDAFORTRAN_FUSED_DFMA \
#                CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED" \
#         NCU_TIMEOUT=600 sbatch jobs/rikyu_subpipe_repeat.sh
#
# The balance axis needs no such job: its MPS sweeps already ran REPEATS=3, and
# the per-round logs give a ratio spread of 0.095% median / 1.42% max on GB200
# (5AE), which is 12x below the 0.71-0.79% gap of the four cells that decide
# that axis' 93.8%.  Those four are real misses; what they are not is
# independent.
set -u
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
export ORDERS="${ORDERS:-63 511 575}"
export PATHS="${PATHS:-CUDAFORTRAN_GEMM CUDAFORTRAN_GEMM_CUTE CUDAFORTRAN_GEMM_FUSED CUDAFORTRAN_FUSED_TC}"
export SHAPES="${SHAPES:-8x8x4}" REPEATS="${REPEATS:-3}" CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-6000}" NCU_TIMEOUT="${NCU_TIMEOUT:-1500}"
export OUTDIR="${OUTDIR:-subpipe_repeat_${SLURM_JOB_ID:-manual}}"
exec bash jobs/subpipe_ncu.sh
