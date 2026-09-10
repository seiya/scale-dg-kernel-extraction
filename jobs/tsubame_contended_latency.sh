#!/bin/bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:20:00
#$ -N dg_h100_clat
# H100 (TSUBAME 4).  See jobs/contended_latency_probe.sh.
export SCALE_DG_ROOT="${SGE_O_WORKDIR:-$PWD}"
export SCALE_DG_MODULES="nvhpc"
export ARCH=sm_90
export OUT="contended_latency_sm90_${JOB_ID:-manual}"
exec bash "$SCALE_DG_ROOT/jobs/contended_latency_probe.sh"
