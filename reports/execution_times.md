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

### 追記 2: occupancy 100% 化と分離可能 lift（commit `103d13b`）

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

### 追記 3: CUDA core 版も occupancy 100% にした（commit `0c65b87`）

`tendency_fused_p7_kernel` はレジスタ 42 本で `Block Limit Registers = 5`、
theoretical occupancy 62.5% だった。`Lift1D` 化と
`attributes(global) launch_bounds(256,8)` で 32 レジスタ / 100% occupancy に
なる。ncu は Slurm job `44039`。同一入力、login node、各 3 回。

| `DqdtKernel_Type` | Main | Cal_tend | CUDA device |
|---|---:|---:|---:|
| `CUDAFORTRAN_FUSED_TC` | **1.415** | **0.890** | fused **0.852** |
| `CUDAFORTRAN_FUSED` | 1.549 | 1.024 | fused 0.986 |

CUDA core 版は `e971ba5` 比 1.17×。両者を 100% occupancy で揃えると
TC 版の優位は 1.35× から **1.16×** に縮む。p=7 の最速は
`CUDAFORTRAN_FUSED_TC` のままである。詳細は
`tc_paper_survey_2407.09621.md` §8。

### 追記 4: `q0 <- q` を stage 1 の RK 更新に融合（本追記を導入したコミット、2026-08-25）

`main.f90` の time-stepping ループは step 先頭で `q0 <- q` の独立カーネルを
launch していた（268.4 MB、`overall_summary_report.md` §8 では 88 µs、
3.05 TB/s）。SSP-RK の stage 1 では定義上 `q0 == q` なので、この保存を
stage 1 の更新カーネルの中に移した。式の形は変えていない
（`rk_a*q0 + rk_b*(q + dt*dqdt)` のまま、`q0` へのストアを 1 行足しただけ）。

- 削減トラフィック: 独立カーネル 268.4 MB/step が消え、既存の更新カーネルに
  134.2 MB のストアが増える。差し引き **-134.2 MB/step**。
- tendency カーネルには一切触れていないので `Cal_tend` と `CUDA device` は不変。

測定は login node、`nstep=1000`、各 3 回の平均。ビルドは
`make CUDA=1 GPUFLAGS=-gpu=cc100`、GB200 1 GPU。

| 入力 / パス | Main（融合前） | Main（融合後） | 差 |
|---|---:|---:|---:|
| p=7 `Ne=32^3` `CUDAFORTRAN_FUSED_TC` | 1.4146 | **1.3115** | -0.1031（-103 µs/step）|
| p=7 `Ne=32^3` `CUDAFORTRAN_FUSED` | 1.5473 | **1.4520** | -0.0953（-95 µs/step）|
| p=255 `Ne=1` `CUDAFORTRAN_GEMM_FUSED` | 4.1889 | **4.0968** | -0.0921（-92 µs/step）|

3 条件とも owned volume point 数は 256^3 で同じなので、削減量が
92-103 µs/step で揃うのは期待どおりである。理論帯域から見積もった
削減（134.2 MB を 5.06 TB/s で 27 µs、消えた 88 µs との差し引きで約 61 µs）
より実測が大きいのは、独立カーネルの launch と、更新カーネルが `q` を
読むときの L2 ヒットが効いているためと考えられる。

数値検証:

- `SCALE_DG_VARYING_COEFF=1`、`Ne=8^3`, `nstep=7`, `CUDAFORTRAN_FUSED_TC` で
  最終 `q(:,1:Ne)` 262,144 点を全点ダンプし、融合前と**ビット一致**。
- `nstep=1000` の min/max 出力（10 行）も、p=7 の 2 パスと p=255
  `CUDAFORTRAN_GEMM_FUSED`、および非 CUDA ビルドの `OPENACC_ASIS`
  （`Ne=8^3`, `nstep=20`, varying coeff）すべてで融合前と一致した。
- 式の並びを変えると FMA の contraction が変わり 15 桁目でずれる。
  最初に `( q0 + dt*dqdt )` と書いた版は 1000 step 後に相対 1e-14 の差が出た。
  **`( q + dt*dqdt )` のまま残すこと**が bit 一致の条件である。

### 追記 5: OpenACC 領域を async にして CUDA カーネルと同じストリームに載せた（本追記を導入したコミット、2026-08-25）

`Main` と `Cal_tend` の差（非 tendency の wall 時間、追記 4 の時点で 422 µs/step）が
非 tendency カーネルの device 時間の合計（約 336 µs/step）より 86 µs/step 大きい
理由を nsys で調べた（Slurm job `44070`、`nstep=60`、p=7 `Ne=32^3`,
`CUDAFORTRAN_FUSED_TC`、`nsys profile --trace=cuda --sample=none --cpuctxsw=none`）。
原因は数値でも帯域でもなく、**カーネルの間で GPU が空いていること**だった。

| stage 内の区間 | 変更前の平均 gap | 変更後 |
|---|---:|---:|
| tendency → RK 更新 | 18.1 µs | 3.9 µs |
| RK 更新 → halo | 13.8 µs | 2.5 µs |
| halo → tendency | 14.4 µs | 3.9 µs |
| **stage 合計** | **46.3 µs** | **10.3–16.8 µs** |
| **1 step (3 stage)** | **138.9 µs** | **50.4 µs** |

変更前の host 側は 1 stage の間に 4 回ブロックしていた。`!$acc parallel loop` に
`async` が無いため nvfortran は launch ごとに `cuStreamSynchronize` を出し
（1000 step で 733 回 → 9 回に減少）、さらに tendency ラッパが device 時間を読むために
毎 stage `cudaEventSynchronize` していた。どちらも「前のカーネルが終わってから
次を launch する」形になるので、launch レイテンシ（API 2–4 µs + launch→開始 3–5 µs）が
毎回むき出しになる。

変更点（数値演算・launch 順序は不変）:

- 時間発展ループの OpenACC 領域（RK 更新、`update_halo`）を `async(ACC_QUEUE)` にし、
  host が結果を読む直前（min/max 出力、`dqdt` ダンプ、ループ終了）だけ `wait` する。
- CUDA Fortran / C++ / cuBLAS / CUTLASS の全カーネルを、その OpenACC キューの
  ストリーム（`acc_get_cuda_stream`）に載せる。同一ストリームなので順序は
  従来と同一に保たれる。
- tendency の device 時間計測（CUDA event）は、同じ呼び出しでは読まず
  **1 回後ろの呼び出しで読む**（イベントを 2 面持ちにした）。値の意味は変わらない。
- OpenACC カーネルで tendency を計算する経路（`OPENACC_ASIS`, `OPENACC_SPLIT`,
  `CUDAFORTRAN_SPLIT`）だけは、その OpenACC 領域が既定キューにあるので
  `advect3d_eq_cal_tend` の先頭で `!$acc wait(ACC_QUEUE)` する。

測定は login node、`nstep=1000`、各 3 回平均。参照は同一ノードで測り直した
`43fe5f2` のビルド。

| 入力 / パス | Main（変更前） | Main（変更後） | 差 | CUDA device（前 → 後）|
|---|---:|---:|---:|---|
| p=7 `Ne=32^3` `CUDAFORTRAN_FUSED_TC` | 1.3099 | **1.2072** | -0.1027（-103 µs/step）| 0.8523 → 0.8518 |
| p=7 `Ne=32^3` `CUDAFORTRAN_FUSED` | 1.4412 | **1.3444** | -0.0968（-97 µs/step）| 0.9845 → 0.9874 |
| p=255 `Ne=1` `CUDAFORTRAN_GEMM_FUSED` | 4.0890 | **3.9601** | -0.1289（-129 µs/step）| 3.6085 → 3.5965 |

device 時間は変わらない（カーネルには触れていない）。残る 50 µs/step は
1 step あたり 9 回の launch の turnaround で、これを消すには CUDA Graph 化のように
launch 自体を減らす必要がある。

**タイマの意味の変化**: host はもう stage ごとに同期しないので、`Cal_tend` と
`Volume derivate + surface lift` は「tendency の wall 時間」ではなく
「host がキューに積んだ device 仕事を待った時間」を含む。カーネル単体の時間は
`CUDA device *`（CUDA event）を見ること。`Main` の意味は変わらない
（ループ直後に `!$acc wait` を入れてある）。

数値検証: `SCALE_DG_VARYING_COEFF=1` で p=7 `Ne=8^3`, `nstep=100` の 5 経路
（`CUDAFORTRAN_FUSED_TC` / `FUSED` / `SPLIT` / `OPENACC_SPLIT` / `OPENACC_ASIS`）と
p=255 `Ne=1`, `nstep=20` の 4 経路（`GEMM_FUSED` / `FUSED_TC` / `GEMM` / `GEMM_CUTE`）で、
stage 1 の `dqdt` 全点ダンプと min/max 系列の**全 18 比較が `43fe5f2` とビット一致**。
非 CUDA ビルド（`make ACC=1` と CPU/OpenMP）も同一結果でビルド・実行できる。

途中で踏んだ失敗も記録しておく。最初は逆向きに
`acc_set_cuda_stream(queue, 0)` で OpenACC キューを CUDA の既定ストリームに
束ねようとしたが、nvfortran は独自のストリームを使い続け（nsys で stream 7 と 14 に
分かれ、halo が tendency の終了 9 µs 前に走っていた）、結果が run ごとに揺れた。
さらに CUTLASS の device 級 GEMM は `gemm_op(args)` の既定引数が
`stream = nullptr` なので、明示的に渡すまで p=255 の GEMM 経路だけ既定ストリームに
残り、5 step 後に 1e-11 のずれとして現れた。**「全部同じストリームに載った」ことは
nsys の stream 列で確認すること。**

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

### 追記 6: p=7 TC の整数・アドレス演算削減と epilogue の転置（本追記を導入したコミット、2026-08-25）

`tc_paper_survey_2407.09621.md` §11 の実測。`tendency_fused_p7_tc_kernel` の
x / y 導関数を shared に置かずアキュムレータのまま使い、`m8n8k4` の出力を
転置して thread が担当するノードを `2*tid`, `2*tid + 1` にした。
SASS の整数命令 176 → 134 本、shared load 36 → 23 本、store 16 → 11 本。
レジスタ 32 本・8 ブロック / SM は維持。

測定は login node、空き GPU 1 枚に固定、`nstep=1000`、版を交互に 12 ラウンド。

| 入力 / パス | CUDA device（前） | CUDA device（後） | 差 |
|---|---:|---:|---:|
| p=7 `Ne=32^3` `CUDAFORTRAN_FUSED_TC` | 0.8518 | **0.8488** | −0.35% |

ncu 単発 launch（job `44819`）では 434.6 → 405.5 µs（−6.7%）と出るが、
実運用の利得はその 1/20 である。理由は §11.5 に記した通りで、
`long scoreboard` stall が 25.5 → 35.0 に増え、律速が発行スロットから
global load レイテンシに移ったためである。

数値検証: `SCALE_DG_VARYING_COEFF=1` で p=7 `Ne=3*4*5`（`Ne=60`）と `Ne=1` の
`dqdt` 全点を `CUDAFORTRAN_SPLIT` と突き合わせ、相対差 1.25e-15 と 1.17e-16。
