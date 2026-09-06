#!/bin/bash
#SBATCH --job-name=dg_l1_roof_ncu
#SBATCH --gpus=1
#SBATCH --time=01:30:00
# The GB200 half of the L1/shared third-term study: jobs/l1_roof_ncu.sh,
# unchanged, under Slurm.  Build for GB200 first
# (make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100), then
#   sbatch jobs/rikyu_l1_roof_ncu.sh
# The retake of reports/a100_report.md 11.7: two clock-control modes x ten
# interleaved repeats x nine configurations.  Job 82908 ran nine invocations in
# one minute, so 180 of them is about 21 minutes; TIME_BUDGET stops the sweep
# at 90 minutes whatever happens, which is what actually bounds the job now
# that the invocation count is no longer nine.
#
# Set REPEATS=1 CLOCKS=base to reproduce job 82908 exactly.
set -u
export REPEATS="${REPEATS:-10}"
export CLOCKS="${CLOCKS:-base none}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/l1_roof_ncu.sh
