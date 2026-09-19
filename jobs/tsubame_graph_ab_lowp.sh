#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -N dg_h100_graph_ab
#$ -j y
# TSUBAME 4 (H100) wrapper for jobs/graph_ab_lowp.sh (CUDA-graph on/off A/B,
# p=7 and 15).  Build with nvhpc/26.1_cuda13.1, cc90 / sm_90, then
#   qsub -g <GROUP> jobs/tsubame_graph_ab_lowp.sh
# Check the node: `qstat -j <id> | grep granted_req` showing gg_mig means a
# MIG node (minutes of start-up per run); qdel and resubmit with -l 'h=!<node>'.
# The two CUTLASS paths run at both 8x8x4 (default) and 16x8x4 (sm_90
# preferred shape), because the paper's Fig. 1(b) and fastest-path table use
# the common shape and the preferred-shape table uses the other.
set -u
export SCALE_DG_MACHINE=h100
export SCALE_DG_RESOURCE="gpu_1=1 (H100 1 GPU)"
export SCALE_DG_ROOT=${SCALE_DG_ROOT:-${SGE_O_WORKDIR:-$PWD}}
cd "$SCALE_DG_ROOT" || exit 1
exec bash jobs/graph_ab_lowp.sh
