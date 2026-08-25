# Execution-time summary

Measured with the CUDA Fortran build (`make CUDA=1 GPUFLAGS=-gpu=cc100`)
on one NVIDIA GB200 compute node. Slurm job `41348` on host `c162`
(`sbatch bench_runs/job_bench_paths.sh`, elapsed 4 min 15 s).
`nstep = 1000` and `DGOptrKernel_OptType = OPT1` for every path.
`dt` follows the corresponding large-case inputs (`1.0e-5` for p=7,
`1.0e-7` for p=255).

Timers are the application reports in seconds. For CUDA Fortran modes,
`CUDA device *` is CUDA-event device time (no host launch/sync).
`Volume derivate + surface lift` is end-to-end wall time for that region.

## p=7, `Ne = 32^3`

| `DqdtKernel_Type` | Main | Cal_tend | Boundary flux | Volume + lift (wall) | CUDA device (detail) |
|---|---:|---:|---:|---:|---|
| `OPENACC_ASIS` | 3.634 | 3.113 | 0.584 | 2.529 | — |
| `OPENACC_SPLIT` | 2.901 | 2.390 | 0.586 | 1.804 | flux 0.425 / deriv 0.645 / lift 0.236 / assemble 0.498 |
| `CUDAFORTRAN_SPLIT` | 2.868 | 2.351 | 0.585 | 1.765 | flux 0.465 / deriv 0.580 / lift 0.213 / assemble 0.472 |
| `CUDAFORTRAN_FUSED` | 1.695 | 1.182 | in fused kernel | 1.182 | fused 1.150 |
| `CUDAFORTRAN_FUSED_TC` | 2.019 | 1.506 | in fused kernel | 1.505 | fused 1.473 |
| `CUDAFORTRAN_GEMM` (`CublasEmulation=.false.`) | 9.271 | 8.743 | in GEMM path | 8.742 | GEMM 8.705 |

Main-time ratio versus `OPENACC_ASIS`:

| Implementation | Main / ASIS |
|---|---:|
| `OPENACC_ASIS` | 1.00 |
| `OPENACC_SPLIT` | 1.25× |
| `CUDAFORTRAN_SPLIT` | 1.27× |
| `CUDAFORTRAN_FUSED` | 2.14× |
| `CUDAFORTRAN_FUSED_TC` | 1.80× |
| `CUDAFORTRAN_GEMM` | 0.39× |

### 追記: `CUDAFORTRAN_FUSED_TC` の再測定（commit `e22dda1`）

上表は commit `299a868` 時点の測定である。`e22dda1` で
`tendency_fused_p7_tc_kernel` の shared memory レイアウトを組み替えて
バンクコンフリクトを除いた結果、p=7 の順位が入れ替わった。
再測定は Slurm job `43618`、同一入力（`Ne=32^3`, `nstep=1000`, `dt=1.0e-5`,
`OPT1`）、同一 GPU。

| `DqdtKernel_Type` | Main | Cal_tend | Volume + lift (wall) | CUDA device |
|---|---:|---:|---:|---:|
| `CUDAFORTRAN_FUSED_TC` (`e22dda1`) | **1.614** | **1.101** | 1.100 | fused **1.068** |
| `CUDAFORTRAN_FUSED` (同ジョブの対照) | 1.692 | 1.182 | 1.181 | fused 1.150 |

ncu 単発カーネル時間では 662.2 µs → 497.2 µs（**1.33×**）で、
CUDA core 版 549.8 µs に対して **1.11×** 速い。
したがって **p=7 の最速パスは `CUDAFORTRAN_FUSED` から
`CUDAFORTRAN_FUSED_TC` に変わった**。経緯と測定値は
`tc_paper_survey_2407.09621.md` §5-6 にある。

### 追記 2: occupancy 100% 化と分離可能 lift（未コミットの作業ツリー）

`e971ba5` の TC カーネルは theoretical occupancy が 75% しかなく
（レジスタ 40 本と smem 28.16 KB がどちらも 6 ブロックで頭打ち）、
`sD*` を `sFlux*` に in-place 化して smem を 15.87 KB に落とし、
`__launch_bounds__(256, 8)` でレジスタを 32 本に抑え、さらに
`Lift_mat`(512×6) を `Lift1D`(8×6) に置き換えた。
測定は login node、同一入力（`Ne=32^3`, `nstep=1000`, `dt=1.0e-5`, `OPT1`）、
各 3 回。ncu は Slurm job `43954` / `43959`。

| `DqdtKernel_Type` | Main | Cal_tend | CUDA device |
|---|---:|---:|---:|
| `CUDAFORTRAN_FUSED_TC`（本追記の変更後） | **1.415** | **0.890** | fused **0.851** |
| `CUDAFORTRAN_FUSED_TC`（`e971ba5`） | 1.646 | 1.128 | fused 1.076 |
| `CUDAFORTRAN_FUSED`（同条件の対照） | 1.714 | 1.190 | fused 1.153 |

device 時間で `e971ba5` 比 **1.26×**、CUDA core 版比 **1.35×**。
ncu 単発カーネル時間は 501.1 µs → 433.1 µs。
ncu はリプレイごとに L2 を流すため実運用より遅く出る（実測は 1 launch
359 µs → 284 µs）ので、両者を混ぜないこと。
経緯・不採用案・残作業は `tc_paper_survey_2407.09621.md` §7 にある。

`CUDAFORTRAN_GEMM` with `CublasEmulation=.true.` did not produce a
timing. The run printed that cuBLAS floating-point emulation APIs are
unavailable and that native FP64 GEMM would be used, then hit the 180 s
timeout.

## p=255, `Ne = 1`

Only fused / GEMM paths are valid for this order.

| `DqdtKernel_Type` | Main | Cal_tend | Volume + lift (wall) | CUDA device |
|---|---:|---:|---:|---:|
| `CUDAFORTRAN_FUSED` | 15.529 | 15.000 | 14.999 | fused 14.966 |
| `CUDAFORTRAN_FUSED_TC` | 13.760 | 13.237 | 13.236 | fused 13.204 |
| `CUDAFORTRAN_GEMM` (`CublasEmulation=.false.`) | 4.439 | 3.906 | 3.905 | GEMM 3.866 |

Main-time ratio versus `CUDAFORTRAN_FUSED` at p=255:

| Implementation | Main / FUSED |
|---|---:|
| `CUDAFORTRAN_FUSED` | 1.00 |
| `CUDAFORTRAN_FUSED_TC` | 1.13× |
| `CUDAFORTRAN_GEMM` | 3.50× |

At p=7 the fused Tensor Core kernel is the fastest complete path as of
commit `e22dda1`; before that rework the plain fused CUDA Fortran kernel
was.  At p=255 the fused kernel is dominated by the large operators, and
`CUDAFORTRAN_GEMM` is the fastest of the three paths measured here
(`CUDAFORTRAN_GEMM_FUSED`, added later, is faster still; see
`overall_summary_report.md` §5).
