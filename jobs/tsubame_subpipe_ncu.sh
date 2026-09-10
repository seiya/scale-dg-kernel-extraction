#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dg_h100_subpipe_ncu
#$ -j y
#
# The H100 half of the SM sub-pipe study: jobs/subpipe_ncu.sh, unchanged, under
# SGE.  H100 is the HOLD-OUT machine -- do not use these rows to choose the
# model form or any coefficient.  They exist to check, after the model is
# fixed on GB200+A100, that the predicted sub-pipe shift (only the DMMA half of
# the fp64+tensor pipe gets cheaper) is what actually happened.
#
#   qsub -g <GROUP> jobs/tsubame_subpipe_ncu.sh
#
# SHAPE is deliberately left at the tree default so that this run matches the
# H100 configuration the published tables use.
set -u
export REPEATS="${REPEATS:-2}"
export CLOCKS="${CLOCKS:-none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export OUTDIR="${OUTDIR:-h100_subpipe_ncu_${JOB_ID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/subpipe_ncu.sh
