#!/bin/bash
#PJM -N dg_a100_roof_bytes
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# The A100 column of jobs/roof_bytes_ncu.sh -- measured DRAM traffic and which
# roof binds, at production clocks.  GB200 is Slurm 83141 and H100 is SGE
# 8585090; both are in reports/measured_dram_and_roofs.md.
#
# Why A100 is worth a third column here and not merely a third machine: the
# two roofs it separates are the ones GB200 cannot separate (its FP64 CUDA-core
# and Tensor-Core peaks are the same number) and H100 separates only weakly
# (its L1 and FP64 ratios against GB200 are 1.199 and 1.197, so a two-machine
# fit cannot tell them apart -- a100_report.md section 10.4).  A100's are 2.058
# and 4.134.  Section 13 of that report already found the binding resource
# differs between GB200 and A100 for ten of sixteen kernels; this job measures
# the traffic and the roofs on the same footing as the other two machines.
#
# Fill in -g, build for cc80 as in jobs/wisteria_paths.sh, submit FROM the
# checkout:
#   module load nvidia/25.9        # NOT with gcc/*: they conflict on Aquarius
#   make clean && make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80
#   pjsub jobs/wisteria_roof_bytes_ncu.sh
#
# ORDERS: p=767 does not fit in 40 GiB (76.2 GiB of payload), so the top order
# is p=575, which fits at 81% (jobs/wisteria_paths.sh).  It is placed last, so
# if ncu's replay buffer pushes it over the card the other three orders are
# already written.  If it does fail, resubmit that order alone with
# REPLAY=application, which costs passes instead of memory.
#
# SHAPE stays 8x8x4: the sm_90 f64 MMA shapes do not exist on A100 and the tree
# refuses them with error stop (a100_prediction.md section 7.2), so 16x8x4
# would only produce failed runs.
#
# WALL CLOCK: share-short caps elapse at 02:00:00.  GB200 ran fourteen
# invocations in 4 min 09 s and H100 in about the same; A100 is 2.5-5x slower
# per path, so twelve invocations should be well inside an hour.  TIME_BUDGET
# stops the sweep at 90 minutes whatever happens.
set -u
export ORDERS="${ORDERS:-7 63 255 575}"
export SHAPE="${SHAPE:-8x8x4}"
export CLOCK="${CLOCK:-none}"
export REPLAY="${REPLAY:-kernel}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export SCALE_DG_MODULES="nvidia/25.9"
export SCALE_DG_MODULE_PURGE=0
export OUTDIR="${OUTDIR:-a100_roof_bytes_ncu_${PJM_JOBID:-manual}}"
ROOT="${SCALE_DG_ROOT:-${PJM_O_WORKDIR:-$PWD}}"
cd "$ROOT" || exit 1
exec bash jobs/roof_bytes_ncu.sh
