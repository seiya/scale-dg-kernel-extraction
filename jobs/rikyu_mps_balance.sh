#!/bin/bash
#SBATCH --job-name=dg_mps_balance
#SBATCH --gpus=1
#SBATCH --time=03:00:00
# GB200 half of the balance sweep.  See jobs/mps_balance.sh for the body; this
# wrapper used to carry its own copy of it (Slurm 96451 ran that copy), and was
# folded onto the shared body on 2026-09-15 so that the in-job %smid record the
# body now takes is made on this machine too.  Same loops, same timers, same
# namelists as the copy, so the columns stay one measurement.
#
#   GB200 at 100% : 25.21 B/SM-clk   (A100 is 10.21, H100 9.17)
#         at  50% : 50.4
#         at  25% : 100.8            -- outside every real machine
#
# The GB200 side of the balance-matched point (jobs/wisteria_mps_match.sh) is
# this machine at 100%:
#   TAG=match_gb200 PCTS=100 ROUNDS=3 sbatch jobs/rikyu_mps_balance.sh
set -u
export SCALE_DG_ROOT=/data1/rkp00015/rku00044/scale-dg-kernel-extraction
export OUTDIR="${OUTDIR:-mps_${TAG:-balance}_${SLURM_JOB_ID:-manual}}"
exec bash "$SCALE_DG_ROOT/jobs/mps_balance.sh"
