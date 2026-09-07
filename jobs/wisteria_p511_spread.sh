#!/bin/bash
#PJM -N dg_a100_p511spread
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g CHANGE_ME
#PJM -j
#PJM -S
#
# measurement_inventory.md 19.7, fourth item: p=511 FUSED_TC is the noisiest
# cell on A100 in both jobs that measured it -- spread 1.32% in the published
# section 2 and clean 3/6 in section 19 -- and no job has asked why.
#
# This asks the cheapest question first: is the spread REPRODUCIBLE, or was it
# two unlucky draws?  Twenty interleaved rounds instead of six give the answer
# without a profiler.  If the spread survives at 20 rounds it is a property of
# the cell and the next step is ncu on A100; if it collapses, the two earlier
# jobs were unlucky and section 19.4's caveat can be tightened.
#
# BEFORE SUBMITTING: build for A100 as jobs/wisteria_paths.sh describes
# (module nvidia/25.9, make CUDA=1 GPUFLAGS=-gpu=cc80 GPUNVCCFLAGS=-arch=sm_80),
# fill in -g above, then:  pjsub jobs/wisteria_p511_spread.sh
#
# One order, five paths, twenty rounds.  At the section 19 A100 times
# (FUSED_TC 32.6 ms/stage, DFMA 83.9) one round is about 75 s, so twenty is
# roughly 25 min; the 2 h cap has room.
set -u
export ORDERS_A="" ORDERS_B="511" ORDERS_C=""
export ROUNDS_HI=20
export VALIDATE=0 TIMING=1
export OUTDIR="${OUTDIR:-a100_p511spread_${PJM_JOBID:-manual}}"
exec bash "${PJM_O_WORKDIR:-$PWD}/jobs/wisteria_paths.sh"
