# Job scripts

Most of this directory is local scratch and is not tracked. The exception is
the **cross-machine measurement harness**, which is committed: a published
machine-to-machine ratio is only reproducible if the script that produced
every column of it is. `.gitignore` carries the allowlist.

## Tracked: the cross-path table

One shared body, one thin wrapper per scheduler. Running the *same* body on
every machine is what makes a ratio one measurement instead of three
conventions -- same namelists, same interleaving, same timer, same median.

| file | role |
|---|---|
| `tsubame_paths.sh` | the shared body. Machine-independent; keeps its original name because `reports/h100_report.md` cites it. |
| `rikyu_paths.sh` | Slurm wrapper (RIKYU, GB200, cc100) |
| `wisteria_paths.sh` | Fujitsu TCS wrapper (Wisteria/BDEC-01 Aquarius, A100 40 GiB, cc80) |
| `tsubame_paths_summarize.py` | median over rounds, per (order, path) |

The body takes its machine-specific parts from the environment, so a new
machine is a new wrapper and no edit to the body:

| variable | default (TSUBAME / RIKYU) | Wisteria |
|---|---|---|
| `SCALE_DG_MODULES` | `nvhpc` | `nvidia/<version>` |
| `SCALE_DG_MODULE_PURGE` | `1` | `0` |
| `SCALE_DG_MACHINE` | `h100` | `a100` |
| `SCALE_DG_PEAK_FP64_CUDA` / `_TENSOR` / `_HBM` | 33.5 / 66.9 / 2.39587 | 9.7 / 19.5 / 1.555 |
| `SCALE_DG_SHAPE_SPECS` | the two `@16x8x4` paths | empty (no sm_90 f64 MMA) |
| `SCALE_DG_RESOURCE` | `gpu_1=1` | the `pjsub` resource line |

The root is found from `SCALE_DG_ROOT`, then `SGE_O_WORKDIR`,
`PJM_O_WORKDIR`, `PWD` -- never from `$0`, which is a spool copy under both
SGE and TCS.

## Tracked: prediction 3

| file | role |
|---|---|
| `cublas_kernel_names_nsys.sh` | which kernels cuBLAS actually runs for the FP64 volume GEMMs, by name |
| `wisteria_cublas_nsys.sh` | TCS wrapper |

Machine-independent as well; it decides `reports/a100_prediction.md`
prediction 3 and prints the verdict.

## Not tracked

One-off ablations, sweeps and `ncu` probes (`ab_*`, `sweep_*`, `time_*`,
`ncu_*`, `nsys_*`). Names are `{tool}_{scope}_{topic}.sh`. Durable profiler
commands belong in `reports/`, not here. Output trees written by these jobs
are not tracked either.

## Submitting

Submit from the repository root; each script finds the checkout and `cd`s
there.

```bash
sbatch jobs/rikyu_paths.sh                 # RIKYU (Slurm)
qsub -g <GROUP> jobs/tsubame_paths.sh      # TSUBAME 4 (SGE)
pjsub jobs/wisteria_paths.sh               # Wisteria (TCS)
```

The two `wisteria_*.sh` wrappers have `#PJM -L rscgrp=` and `#PJM -g` left as
`CHANGE_ME`; fill them in before submitting. Build instructions per machine
are in each wrapper's header.
