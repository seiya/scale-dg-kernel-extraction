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

`CUDAFORTRAN_GEMM` with `CublasEmulation=.true.` did not produce a timing in
this historical run. The later investigation in
`cublas_emulation_survey.md` showed that the API-availability message was
incorrect: an enum value had been tested with `defined()`, while environment
variables still enabled `EAGER` FP64 emulation. The run was progressing about
about 131 times more slowly than native FP64 at p=7, so `nstep=1000` was unsuitable.
Use `nstep=1--10` for this comparison.

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

### 追記 7: 1 step を CUDA Graph 化した（本追記を導入したコミット、2026-08-25）

追記 5 でカーネル間の GPU アイドルを 139 → 50 µs/step まで下げた後、残りは
1 step 9 回の launch turnaround そのものだった。`overall_summary_report.md`
§12 の項目 5 に挙げていた **CUDA Graph 化**を実装した。

SSP-RK3 の 1 step（halo 更新・tendency・RK 更新 × 3 stage）を
`cudaStreamBeginCapture` / `cudaStreamEndCapture` で 1 回だけ捕捉し、以降の
step は `cudaGraphLaunch` で再生する。**カーネル・引数・実行順序・データは
一切変えていない**。捕捉は 2 step 目で行う（capture は何も実行しないので、
host が結果を読む 1 step 目は直接 launch する）。namelist の
`UseCudaGraph = .true.` で有効。

再生は Fortran のラッパを通らないので、tendency の CUDA event による device
時間はこのモードでは採れない。`Cal_tend` と内訳は `not measured (graph)` と
表示され、測るのは wall 時間だけになる。

測定は login node、GB200 1 枚、`nstep=1000`、`make CUDA=1`、
graph on / off を交互に実行。値は `Main`（秒）。

| 入力 / パス | graph なし | graph あり | 差 | µs/step |
|---|---:|---:|---:|---:|
| p=7 `Ne=32^3` `CUDAFORTRAN_FUSED_TC` | 1.2038 | **1.1716** | −2.7% | −32 |
| p=7 `Ne=32^3` `CUDAFORTRAN_FUSED` | 1.3441 | **1.3104** | −2.5% | −34 |
| p=255 `Ne=1` `CUDAFORTRAN_GEMM_FUSED` | 3.9730 | **3.8545** | −3.0% | −119 |
| p=255 `Ne=1` `CUDAFORTRAN_GEMM_CUTE` | 4.2646 | **4.1328** | −3.1% | −132 |
| p=255 `Ne=1` `CUDAFORTRAN_FUSED` | 15.346 | **15.288** | −0.4% | −59 |

各ラウンドの生値（p=7 TC）: graph なし 1.2031 / 1.2109 / 1.2033 / 1.1978、
graph あり 1.1742 / 1.1679 / 1.1729 / 1.1713。

イベント計測自体のコストも分離できる。p=7 TC で graph なし・イベント off
（`MeasureKernelTime = .false.`）は 1.199 秒で、graph なし・イベント on の
1.204 秒との差 5 µs/step がイベントの host コストである。graph の利得
32 µs/step のうち 27 µs/step はイベントとは無関係な launch turnaround の分。

p=255 `CUDAFORTRAN_FUSED` の利得が小さいのは、この経路が 1 step 15.3 ms と
カーネル時間に支配されていて、同じ 60 µs 前後の launch コストが相対的に
埋もれるためである。**絶対量はどのパスでもほぼ同じ**で、削れた launch
コストは 32〜132 µs/step の範囲にある。

#### nsys で見たカーネル間の隙間（Slurm job `45686`, `nstep=60`, p=7 `FUSED_TC`）

追記 5 と同じ手順で graph 無しの 1 stage を追うと、定常部 382 カーネル
（約 42 step 分）の span 51.07 ms に対し busy 49.55 ms、隙間の合計 1.511 ms
である。うち 0.211 ms は `output_interval` の min/max reduction 前後の 1 回
限りの待ちなので、それを除くと **30.6 µs/step** が step ごとの launch
turnaround になる。内訳は 1 stage あたり halo → tendency 3.97 µs、
tendency → RK 更新 3.9〜4.1 µs、RK 更新 → 次の halo 2.3 µs である。

end-to-end で測った graph の利得 32 µs/step は、この 30.6 µs/step とほぼ
一致する。**CUDA Graph はカーネル間の隙間をほぼ全部消している。**

graph 側のトレースは採れなかった。**この環境では nsys を graph 再生パスに
当てるとハングする**（Slurm job `45707`, nsys 2026.1.1（NVHPC 26.3 同梱）, GB200）。
`timeout` で囲んで 3 通り試した結果は次のとおりで、同じバイナリでも
`UseCudaGraph = .false.` なら nsys は正常に採れる（job `45686`）。

| 試行 | 結果 |
|---|---|
| プロファイラ無し | `rc=0`、30 step が 35 ms で完走 |
| `nsys profile`（既定の graph 粒度）| **`rc=124`（180 s で timeout）**、プログラム出力なし |
| `nsys profile --cuda-graph-trace=node` | **`rc=124`（180 s で timeout）**、同上 |

いずれも `.nsys-rep` は生成されるが `nsys stats` は
`PROCESSED (EMPTY RESULTS)` で GPU trace を含まない。したがって上の結論は、
graph 無しの隙間 30.6 µs/step と end-to-end の利得 32 µs/step が一致すること
による。graph 側の内訳が要るときは、nsys ではなく CUDA event を graph 内に
仕込むなど別の手段が必要である。

なお RIKYU で nsys を投げる際は `export DEBUGINFOD_URLS=` と
`--resolve-symbols=false` の両方が必須で、欠けるとジョブが終わらない。

数値検証: `SCALE_DG_VARYING_COEFF=1`（`u,v,w,Escale,normal_fn,Fscale` を全点で
変化させる）、p=7 `Ne=4*5*3`（`Ne=60`, 30720 点）、`nstep=300` の後の `q`
全点を比較（`SCALE_DG_DUMP_Q`）。

| パス | graph on と off の最大差 | `FUSED_TC` との相対差 |
|---|---:|---:|
| `CUDAFORTRAN_FUSED_TC` | 0（ビット一致）| — |
| `CUDAFORTRAN_FUSED` | 0（ビット一致）| 1.45e-15 |
| `CUDAFORTRAN_SPLIT` | 0（ビット一致）| 1.45e-15 |

p=255 は `Ne=1` と `Ne=2` の両方で、`CUDAFORTRAN_FUSED` /
`CUDAFORTRAN_FUSED_TC` / `CUDAFORTRAN_GEMM` / `CUDAFORTRAN_GEMM_FUSED` /
`CUDAFORTRAN_GEMM_CUTE` の 5 経路とも graph on / off で `q` の min / max が
完全一致した。

### 追記 8: p=255 の separable lift を 1 本のカーネルにした（2026-08-25）

p=255 の lift は `Lift1D(Nq,6)` と 6 枚の面から作る 3 つの rank-2 項の和で、
これまでは pack 2 本 + copy 1 本のカーネルと `K=2` の cuBLAS GEMM 3 本で
組んでいた。3 本の GEMM は `beta=1` で `lift_out` に**累積**するため、
`Ne=1`, `Np=256³`（`lift_out` = 134 MB）では

| | traffic |
|---|---:|
| x-lift GEMM（write） | 134 MB |
| y-lift GEMM（read+write） | 268 MB |
| z-lift GEMM（read+write） | 268 MB |
| **lift GEMM 群 計** | **670 MB** |

を流していた。176 µs / tendency call に対して約 3.8 TB/s であり、
このカーネル群は演算ではなく `lift_out` の往復そのものが律速だった。
面データは 6 × `nq2` × 8 B = 3 MB しかないので、3 項を 1 スレッドで
まとめて評価して `lift_out` を **1 回だけ書く** `separable_lift_kernel` に
置き換えた（write 134 MB のみ）。加算順は GEMM 3 本と同じ `(x+y)+z` に
そろえてある。

測定は login node、`nstep=1000`、`UseCudaGraph = .false.`、
`bench_runs/p255_*.conf`（`Ne=1`, `PolyOrder=255`, `dt=1.0e-7`, `OPT1`）、
各 2–3 回の代表値。

| `DqdtKernel_Type` | 変更前（`514853f`） | 変更後 | |
|---|---:|---:|---:|
| `CUDAFORTRAN_GEMM_FUSED` | 3.971 | **3.635** | −8.5% |
| `CUDAFORTRAN_GEMM_CUTE` | 4.279 | 3.954 | −7.6% |
| `CUDAFORTRAN_GEMM` | 4.241 | 3.962 | −6.6% |

数値検証: `SCALE_DG_VARYING_COEFF=1` で `dqdt` 全点を比較（`SCALE_DG_DUMP_DQDT`）。

| 比較 | max_abs_diff |
|---|---:|
| p=7 `Ne=8³` `CUDAFORTRAN_GEMM` 変更後 vs 変更前 | 0（ビット一致）|
| p=7 `Ne=8³` `CUDAFORTRAN_GEMM` 変更後 vs `CUDAFORTRAN_SPLIT` | 0（ビット一致）|
| p=255 `Ne=1` `CUDAFORTRAN_GEMM_FUSED` 変更後 vs 変更前 | 0（ビット一致）|
| p=255 `Ne=1` `CUDAFORTRAN_GEMM_FUSED` 変更後 vs `CUDAFORTRAN_FUSED_TC` | 3.553e-15 |

`§5.1` の内訳表（job 43219）は変更前の値である。lift 176 µs と
pack/copy 9.8 µs がこの変更の対象で、`lift_out` を読む z-epilogue 側の
134 MB はまだ残っている。

### 追記 9: p=255 の lift を z-epilogue に畳み込んだ（2026-08-25）

追記 8 で `lift_out` の生成は 1 パスになったが、z-epilogue はまだその
134 MB を読んでいた。z GEMM のユーザ問題は `(m=Nq², n=Nq)` の column-major
なので CUTLASS は転置した row-major 問題として解く。つまり **epilogue タイルの
row が z 添字 `k`、column が xy 面添字 `p = i + j*Nq`** であり、6 面
（計 3 MB）から lift をその場で組み立てられる。`lift_out` は完全に消えた。

素直に書いた最初の版は `Main 3.635 → 3.615` （−0.6%）にしかならなかった。
epilogue が出力 1 点ごとに `p % Nq` / `p / Nq` の整数除算をしていたためで、
z GEMM は SM throughput 79.8% の演算律速だからここが効く。
`PredicatedTileIterator::operator++` は `thread_start_row_` しか進めない
ので、column に依存する量（`i`, `j`, 除算、z 面の 2 値、`Lift1D` の 4 値）は
`kIterations` ループを通じて不変である。これらをループ外に括り出した版が
採用値である。

測定は login node、`nstep=1000`、`bench_runs/p255_gemm_fused.conf`、
graph off。GPU が共有のため外れ値が出るので最小値付近の代表値。

| 版 | Main | `CUDA device GEMM fused` |
|---|---:|---:|
| `514853f`（3 本の lift GEMM） | 3.971 | — |
| 追記 8（`separable_lift_kernel`） | 3.635 | 3.266 |
| lift を epilogue へ（除算そのまま） | 3.615 | 3.244 |
| **lift を epilogue へ（column 不変量を hoist）** | **3.463** | **3.081** |

device 時間は 3000 tendency call あたりなので 1089 → 1027 µs/call。
`514853f` からの通算は Main **−12.8%**。
`CUDAFORTRAN_GEMM` / `CUDAFORTRAN_GEMM_CUTE` は z-epilogue を使わないので
追記 8 のまま（3.946 / 3.960）。

`UseCudaGraph = .true.` でも動作する（Main 3.377）。

数値検証: `SCALE_DG_VARYING_COEFF=1`、`dqdt` 全点比較。

| 比較 | max_abs_diff |
|---|---:|
| p=255 `Ne=1` epilogue lift vs 追記 8 のカーネル lift | 0（ビット一致）|
| p=255 `Ne=1` epilogue lift vs `CUDAFORTRAN_FUSED_TC` | 3.553e-15 |
| p=255 `Ne=2` `GEMM_FUSED` vs `CUDAFORTRAN_FUSED_TC` | 3.553e-15 |
| p=7 `Ne=8³` `CUDAFORTRAN_GEMM` vs `CUDAFORTRAN_SPLIT` | 0（ビット一致）|

非 CUDA ビルド（`make clean && make`）も通ることを確認した。

### 追記 10: `volume_flux_kernel` のロードをストアより前にまとめた（2026-08-25）

追記 9 の時点で GEMM 系に残る帯域律速の独立カーネルは `volume_flux_kernel`
だけだった。commit `d7b1853` の実行ファイルを ncu で測ると、トラフィックは
理論最小そのものだが**どのユニットも飽和していない**ことが分かった
（job 46163、`p=255 Ne=1`、`CUDAFORTRAN_GEMM_FUSED`）。

| 指標 | 実測 | 理論 |
|---|---:|---:|
| `dram__bytes_read.sum` | 536.9 MB | 536.9 MB（`q,u,v,w`）|
| `dram__bytes_write.sum` | 364.2 MB | 402.7 MB（`flux_x/y/z`、差は L2 残留）|
| ld / st セクタ効率 | 100% / 100% | |
| `smsp__inst_executed_op_global_ld.sum` | 3.15 M = **6 / warp** | 2.10 M = 4 / warp |
| DRAM / L1 / L2 / SM throughput | 65.7 / 52.5 / 36.8 / 33.7 % | |

`flux_x(idx) = q(idx)*u(idx)` を 3 行並べた書き方だと、nvfortran はロードを
ストアと交互に発行し、`q` を 3 回読み直していた。4 本のロードを
**ストアより前にまとめて**発行させると次のようになる（job 46183）。

| | 変更前 | 変更後 |
|---|---:|---:|
| global ld 命令 / warp | 6 | **4** |
| ncu duration | 173.2 µs | **135.0 µs** |
| DRAM throughput | 65.7% | **83.4%** |
| L1 / SM throughput | 52.5 / 33.7% | 67.0 / 43.2% |
| register / occupancy | 20 / 84.1% | 24 / 79.9% |
| **nsys duration** | **150.6 µs** | **125.9 µs**（−16.4%）|

893 MB / 125.9 µs = **7.09 TB/s**、参照ピーク 7.9 TB/s の **90%**。
`q` だけをレジスタに退避した版は 3.080 s のままで**まったく効かない**。
効くのはストアを挟まないロード窓のほうである。1 スレッド 2 / 4 / 8 点に
増やす版も試したが、2 点は同値、4 / 8 点はわずかに悪化した。

`nstep=1000`、login node、graph off。`CUDAFORTRAN_FUSED` 系はこのカーネルを
使わないので変化しない。

| path | 変更前 | 変更後 | |
|---|---:|---:|---:|
| p=255 `CUDAFORTRAN_GEMM_FUSED` | 3.4469 | **3.3702** | −2.2% |
| p=255 `CUDAFORTRAN_GEMM` | 3.9403 | 3.8713 | −1.8% |
| p=255 `CUDAFORTRAN_GEMM_CUTE` | 3.9539 | 3.8820 | −1.8% |
| p=7 `CUDAFORTRAN_SPLIT` | 2.7172 | **2.6440** | −2.7% |
| p=7 `CUDAFORTRAN_FUSED` / `FUSED_TC` | 1.3464 / 1.2036 | 1.3452 / 1.2043 | ±0 |

device 時間（`CUDA device GEMM fused`、3000 call）は 3.0812 → 3.0058 s、
1027 → 1002 µs/call。`FUSED volume GEMM only` は 2.541 s で不変なので、
差分はすべてこのカーネルに帰属する。

数値検証: `SCALE_DG_VARYING_COEFF=1`、`dqdt` 全点比較（`SCALE_DG_DUMP_DQDT`）。

| 比較 | max_abs_diff |
|---|---:|
| p=255 `Ne=1` `GEMM_FUSED` 変更後 vs 変更前 | 0（ビット一致）|
| p=255 `Ne=1` `GEMM` 変更後 vs 変更前 | 0（ビット一致）|
| p=255 `Ne=1` `GEMM_CUTE` 変更後 vs 変更前 | 0（ビット一致）|
| p=7 `Ne=8³` `GEMM` 変更後 vs 変更前 | 0（ビット一致）|
| p=7 `Ne=8³` `CUDAFORTRAN_SPLIT` 変更後 vs 変更前 | 0（ビット一致）|

### 追記 11: 帯域律速カーネルを 2 本目のストリームで GEMM の裏に隠した（2026-08-25）

`overall_summary_report.md` §12-7 の案を実測した。x GEMM（249.7 µs、SM 87.9%、
DRAM 10.1%）と y GEMM（243.5 µs、SM 89.2%、DRAM 6.3%）が回っている 493 µs の
あいだ DRAM は空いているので、そこに帯域律速のカーネルを流し込む。
方向ごとの依存は独立で、x GEMM が要るのは `flux_x` だけ、z GEMM が要るのは
`flux_z` と `flux_bnd` だけである。

3 つの分け方を測った（`p255_gemm_fused.conf`、`nstep=1000`、graph off、
login node、値は `CUDA device GEMM fused` 秒 / 3000 call の µs）。

| 版 | main stream | side stream | join | device | µs/call |
|---|---|---|---|---:|---:|
| 変更前 | 全部 | — | — | 3.0058 | 1001.9 |
| A | `flux_x` | `flux_y,z` + elembnd | y GEMM の前 | 3.0245 | 1008.2 |
| A' | A の fork を `flux_x` の後ろへ | 同上 | 同上 | 3.0550 | 1018.3 |
| B | `flux_x,y` | `flux_z` + elembnd | z GEMM の前 | 3.0034 | 1001.1 |
| **C** | volume flux 全部 | **elembnd のみ** | z GEMM の前 | **2.9646** | **988.2** |

**A と B が効かない理由はレジスタである。** x GEMM は 212 reg × 128 thread の
CTA が 2 個/SM で 54,272 / 65,536 本を占め、**SM あたり 11,264 本しか空いていない**。
24 reg の flux ブロック（256 スレッド = 6,144 本）は 1 個しか同居できないので、
side stream 側の並列度は SM あたり 256 スレッド、単独実行時の約 1/8 になる。
`flux_y,z` の 671 MB を 250 µs の窓で流すには 2.7 TB/s 要るが、その並列度では
出ない。結果として隠れず、しかも DRAM を奪って GEMM 側を遅くする
（A' は xy GEMM が +68 µs/call）。105 MB・19.6 µs の elembnd なら入る。

C の重なりは nsys で直接確認した（job 46362、`nstep=20`）。

```
strm=14  start=3436141.0us  dur=253.0us  end=3436394.0us  x GEMM
strm=15  start=3436359.3us  dur= 26.9us  end=3436386.2us  elembnd_flux_kernel
```

elembnd は x GEMM の区間に**完全に収まっている**。代償は elembnd 自身が
19.6 → 26.9 µs（+37%）、x GEMM が 249.7 → 253.0 µs（+1.3%）で、差し引き
−15 µs/call。開始が x GEMM の 218 µs 後であることから、ブロックは GEMM の
最後の wave の隙間に入っていると分かる。

**CUDA Graph の replay では逆に損になる。** 同じ構造を捕捉して再生すると
+5 µs/call で、graph on の Main は 3.2877 → 3.3104 と悪化した。そこで
`UseCudaGraph` のときは `cuda_dg_set_side_stream(.false.)` で 1 本に戻す
（`main.f90`）。無効時は elembnd を volume flux の**前**に戻すことも必要で、
これを怠ると graph on が +0.6% になる。elembnd は `VMapM`/`VMapP` 経由で `q` を
gather するので、volume flux が `q,u,v,w` を L2 に流し込む前のほうが安い。

`nstep=1000`、login node。

| path | 変更前 | 変更後 | |
|---|---:|---:|---:|
| p=255 `GEMM_FUSED`（graph off） | 3.3723 | **3.3293** | −1.3% |
| p=255 `GEMM_CUTE` | 3.8820 | 3.8368 | −1.2% |
| p=255 `GEMM` | 3.8713 | 3.8613 | −0.3% |
| p=255 `GEMM_FUSED`（graph on） | 3.2877 | 3.2892 | ±0（無効化） |
| p=7 `CUDAFORTRAN_GEMM` | 5.487 | 5.486 | ±0（無効） |

p=7 の `CUDAFORTRAN_GEMM` では side stream を使わない。そこでは elembnd が
181 µs、volume GEMM は 8×8 行列の 24 launch なので、隠す先の窓より隠すものの
ほうが大きく、そのまま重ねると **+3.1%** になることを実測した
（`Nq == CUDA_P255_NQ` で切り分けている）。

数値検証: `SCALE_DG_VARYING_COEFF=1`、`dqdt` 全点比較。

| 比較 | max_abs_diff |
|---|---:|
| p=255 `Ne=1` `GEMM_FUSED` 変更後 vs 変更前 | 0（ビット一致）|
| p=255 `Ne=1` `GEMM_FUSED` graph on 変更後 vs 変更前 | 0（ビット一致）|
| p=255 `Ne=1` `GEMM` / `GEMM_CUTE` 変更後 vs 変更前 | 0（ビット一致）|
| p=7 `Ne=8³` `GEMM` 変更後 vs 変更前 | 0（ビット一致）|

非 CUDA ビルド（`make clean && make`）も通ることを確認した
（stub に `cuda_dg_set_side_stream` を追加）。

### 追記 12: 全カーネルのロード命令数を監査し、SSP-RK 更新を 1 次元ループにした（2026-08-25）

追記 10 で `volume_flux_kernel` が 1 warp あたり 6 命令の global load を出して
いた（あるべきは 4）ことが分かったので、**同じ欠陥が他に無いかを全カーネルで
測った**。指標は `smsp__inst_executed_op_global_ld.sum ÷ warp 数` で、これを
ソース上の相異なるロード数と突き合わせる（job 46402 / 46417）。

| kernel | 実測 ld/warp | ソース最小 | 判定 |
|---|---:|---:|---|
| `volume_flux_kernel`（追記 10 後） | 4.00 | 4 | ○ |
| `elembnd_flux_kernel` | 14.00 | 14 | ○（`normal_fn` の 3 本は VelM/VelP で共有されている）|
| `separable_lift_kernel` | 12.00 | 12 | ○ |
| `dqdt_assembly_kernel` | 7.00 | 7 | ○ |
| `volume_deriv_p7_kernel` | 5.00 | 3.25 | 述語化された `D1D`/`D1D_tr` の 2 命令が全 warp で発行される。sector は 26/warp = 理論どおりで**メモリトラフィックは無い** |
| `surface_lift_p7_kernel` | 12.00 | 12 | ○ |
| `tendency_fused_p7_kernel` | 35.50 | 35.5 | ○（`q(idx1)` を 3 回書いても shared store を挟むだけなので CSE される）|
| `tendency_fused_p7_tc_kernel` | 32.50 | 32.5 | ○（`Escale` が `double2`）|
| `update_halo` | 2.00 | 2 | ○ |
| **SSP-RK 更新 stage 1** | **4.00** | 2 | **×** |
| **SSP-RK 更新 stage≥2** | **5.00** | 3 | **×** |

CUDA Fortran のカーネルはすべて最小だった。**外れたのは時間発展ループ側の
OpenACC カーネル 2 本だけ**である。余分な 2 本は `rk_a(stage)` と `rk_b(stage)`
で、`stage` が実行時値なので配列参照が 1 スレッドあたり 1 ロードになっていた。

ところが**それをホスト側でスカラーに読み出しても −0.2〜0.5% にしかならない**
（p=255 `GEMM_FUSED` 3.329 → 3.324、p=7 `FUSED_TC` 1.208 → 1.201）。
2 本とも全スレッドが同じアドレスを読むので L1 に当たり、費やしているのは
発行スロットだけだからである。

**本当の律速は `collapse(2)` だった。** ncu では DRAM 41.7% に対し
SM throughput が 62.7% と高く、3 ロード 1 ストア 3 演算のカーネルとしては
説明がつかない。`collapse(2)` は平坦化したスレッド番号から 2 つの添字を
復元するために**実行時値 `Np` による整数除算**を要求する（追記 9 で
z-epilogue に対して得たのと同じ罠である）。owned 領域 `q(:,1:Ne)` は連続なので、
1 次元ループに書き直せば除算そのものが消える。

| | 変更前（`collapse(2)`） | 変更後（1 次元） |
|---|---:|---:|
| ld/warp（stage 1 / stage≥2） | 4.00 / 5.00 | **2.00 / 3.00** |
| ncu duration（stage 1 / stage≥2） | 152.5 / 154.1 µs | **91.2 / 86.7 µs** |
| SM throughput | 62.7% | **35.3%** |
| DRAM throughput | 41.7% | **74.2%** |
| DRAM バイト | 479.7 / 509.4 MB | 479.2 / 509.8 MB（不変）|

演算律速から帯域律速に変わった。1 step あたり stage 1 が 1 回、stage≥2 が
2 回なので、カーネル時間は約 321 → 185 µs/step になる。

`nstep=1000`、login node、3 ラウンド交互測定の代表値。**tendency に手を
入れていないので、全パスが同じだけ得をする。**

| path | graph | 変更前 | 変更後 | |
|---|---|---:|---:|---:|
| p=7 `CUDAFORTRAN_FUSED_TC` | off | 1.208 | **1.131** | −6.4% |
| p=7 `CUDAFORTRAN_FUSED_TC` | on | 1.171 | **1.083** | −7.5% |
| p=7 `CUDAFORTRAN_FUSED` | off | 1.345 | 1.250 | −7.0% |
| p=7 `CUDAFORTRAN_SPLIT` | off | 2.65 | 2.55 | −3.6% |
| p=7 `OPENACC_ASIS` | off | 3.50 | 3.41 | −2.6% |
| p=255 `CUDAFORTRAN_GEMM_FUSED` | off | 3.329 | **3.232** | −2.9% |
| p=255 `CUDAFORTRAN_GEMM_FUSED` | on | 3.289 | **3.192** | −3.0% |
| p=255 `CUDAFORTRAN_GEMM` | off | 3.861 | 3.763 | −2.5% |
| p=255 `CUDAFORTRAN_GEMM_CUTE` | off | 3.837 | 3.741 | −2.5% |
| p=255 `CUDAFORTRAN_FUSED_TC` | off | 13.56 | 13.57 | ±0（自カーネルが 13 秒）|

数値検証: `SCALE_DG_VARYING_COEFF=1` で `q` 全点を比較（`SCALE_DG_DUMP_Q`）。

| 比較 | 結果 |
|---|---|
| p=255 `Ne=1` `GEMM_FUSED` 変更後 vs 変更前 | ビット一致 |
| p=7 `Ne=8³` `FUSED_TC` / `FUSED` / `GEMM` 変更後 vs 変更前 | ビット一致 |
| p=7 `Ne=8³` `FUSED_TC` `nstep=200` 変更後 vs 変更前 | ビット一致 |
| 同上 graph on vs graph off | ビット一致 |
| CPU/OpenMP ビルド（`OPENACC_ASIS`）vs GPU | max_abs_diff 2.7e-20 |

---

### 追記 13: p=255 z GEMM の shared store バンクコンフリクトを消した（2026-08-26）

`tma_survey.md` §2.2 が残していた標的。`cuda_cutlass_gemm_fused.cu` の z GEMM で
epilogue のアキュムレータ smem タイルの padding を 4 → 8 doubles にし、
行ストライドを 36 → 40 doubles（= バンク 16 本ぶん）にした。原因と帰属の訂正は
`overall_summary_report.md` §8.9 を参照。

測定は login node、`bench_runs/p255_gemm_fused.conf`（`nstep=1000`、graph off）、
版を交互に 3 ラウンド。ビルドは `make CUDA=1 GPUFLAGS=-gpu=cc100`。

| 版 | Main | `CUDA device GEMM fused` | `FUSED volume GEMM only` |
|---|---:|---:|---:|
| 前（`9eff7f8`） | 3.2315 | 2.9647 | 2.5559 |
| 後 | 3.2313 | 2.9642 | 2.5551 |

**時間は動かない**（差はラウンド間ばらつきの内側）。ncu では
shared store のバンクコンフリクトが 1,165,739 → 136,379（−88%）、
store wavefronts が 2.21 M → 1.18 M（理想 1.05 M）まで落ちているので、
消えていないのではなく、このカーネルが shared 経路で律速されていない。
Slurm job `49543` / `49546`。

数値検証:

| 比較 | 結果 |
|---|---|
| p=255 `Ne=1` `GEMM_FUSED` 変更後 vs 変更前（`SCALE_DG_DUMP_DQDT`） | ビット一致 |
| p=255 `Ne=1` `GEMM_FUSED` vs `GEMM`（`SCALE_DG_VARYING_COEFF=1`） | 相対 4.16e-16 |
| p=255 `Ne=2` `GEMM_FUSED` vs `GEMM`（同上） | 相対 4.16e-16 |

---

### 追記 14: p=7 の ±x 面 M 側 gather を shared 経由にした（2026-08-26）

`cuda_dg_kernels_tc.cu` の `tendency_fused_p7_tc_kernel` で、面 2 と 4
（±x 法線）の M 側 `q,u,v,w` を volume ロード時に shared へ退避し、
face セクションから読む。global ロードセクタ 95.9 M → 78.6 M（−18%）。
帰属の測り方と外した 3 版は `tc_paper_survey_2407.09621.md` §13、
まとめは `overall_summary_report.md` §13.2。

測定は login node、`nstep=1000`、版を交互に 4 ラウンド。ビルドは
`make CUDA=1 GPUFLAGS=-gpu=cc100`。

| path | graph | 変更前（`63a4234`）| 変更後 | |
|---|---|---:|---:|---:|
| p=7 `CUDAFORTRAN_FUSED_TC` Main | off | 1.1092 | **1.0656** | −3.9% |
| p=7 `CUDAFORTRAN_FUSED_TC` Main | on | 1.0738 | **1.0367** | −3.5% |
| p=7 `CUDA device fused tendency` | off | 0.8497 | **0.8060** | **−5.1%** |

他のパスと p=255 は同じカーネルを通らないので不変
（p=255 `GEMM_FUSED` の `q` はビット一致で確認）。

数値検証（`SCALE_DG_VARYING_COEFF=1`）:

| 比較 | 結果 |
|---|---|
| p=7 `Ne=2³` `FUSED_TC` 変更後 vs `63a4234`（owned `dqdt` 全点）| ビット一致 |
| p=7 `Ne=4³` 同上 | ビット一致 |
| p=7 `Ne=2³` `FUSED_TC` vs `CUDAFORTRAN_SPLIT` | max_abs_diff 1.78e-15 |
| p=7 `Ne=4³` 同上 | max_abs_diff 2.86e-14 |
| p=255 `Ne=1` `GEMM_FUSED` 変更後 vs 変更前（`q` 全点）| ビット一致 |

CUDA ビルドと非 CUDA ビルドの両方が通ることを確認済み。

### 追記 15: p=255 の RK 更新を z epilogue に融合するのは損（不採用、2026-08-26）

`dqdt` のストアと読み戻し（ステージあたり 268 MB）を消すために、SSP-RK の
ステージ更新を z GEMM の epilogue に入れた。namelist `FuseRKUpdate` で
切り替える形で実装し、`nstep=1000` で測った（3 ラウンド）。

| | Main |
|---|---:|
| `FuseRKUpdate = .false.` | **3.228 s** |
| `FuseRKUpdate = .true.` | 3.407 s（**+5.5%**）|

Slurm job `49674` の nsys でカーネル単位に分けると、z GEMM が
338.8 → 481.1 µs（stage≥2）/ 452.0 µs（stage 1）に膨らみ、RK カーネル
225.6 µs/step を消しても差し引き +172 µs/step になる。理由は帯域ではなく
4 ブロック/SM で隠せないロードレイテンシで、`overall_summary_report.md`
§13.3 に分解がある。**コードは差し戻した。**

数値は融合版も正しく、`FuseRKUpdate` の on / off で `q` 全点が
相対 2.22e-16（1 ulp）で一致した。ビット一致にはならず、更新式を
4 通り（FMA を両向きに明示、契約なし）に書き分けても同じ 1 ulp だったので、
差は更新の算術ではなく、epilogue の融合版・非融合版でテンデンシー式の
スケジューリングが変わることに由来する。

### 追記 16: p=7 経路横断の再測定（2026-08-29）

p=7 専用の gap study は無いので、ここに現行ツリーの横断測定を置く。
commit `2dadc41`、login node GPU 1、3-run 中央値。入力は `conf_perf_p7.conf`
（`Ne=32³`、`nstep=20`、graph off）の `DqdtKernel_Type` だけを差し替え。
µs/stage は `CUDA device *`（SPLIT は 4 本 + elembnd、OpenACC は volume wall +
elembnd）。冒頭の `nstep=1000` 表と、追記の `FUSED_TC` 再測定は当時の値のまま残す。

| 経路 | Main [ms/step] | µs/stage |
|---|---:|---:|
| `OPENACC_ASIS` | 3.492 | 1065.8 |
| `OPENACC_SPLIT` | 2.708 | 807.8 |
| `CUDAFORTRAN_SPLIT` | 2.565 | 764.1 |
| `CUDAFORTRAN_FUSED_DFMA` | 1.528 | 427.8 |
| **`CUDAFORTRAN_FUSED_TC`** | **1.073** | **274.9** |
| `CUDAFORTRAN_GEMM` | 5.088 | 1635.4 |

**最速は `CUDAFORTRAN_FUSED_TC` のまま。** 冒頭表の GEMM Main 9.271 s（nstep=1000）
は RK 最適化前で、同一 conf では 5.088 ms/step。

この節の `CUDAFORTRAN_FUSED` 行は iso-schedule DFMA（`UseTc=false`）の測定である。
経路名は `CUDAFORTRAN_FUSED_DFMA` と読む。旧 Fortran 融合（device 〜324 µs）は
CC 最適の旧測であり、DFMA の 427.8 µs とは並べない。FLOP/s と DRAM は
[`README.md`](README.md) のまとめ表。

### 追記 17: p=7 TC の z 往復天井（2026-08-29、不採用）

`sDz` のストア/ロードとバリア 2 本を消す不正アブレーション。数値は壊れる。
device 中央値 **274.14 → 270.50 µs/stage（−1.3%）**。天井が 4 µs なので
写像を直す実装はしない。詳細は `tc_paper_survey_2407.09621.md` §16。
コードはベースに戻してある。

### 追記 18: p=7 CUDA-core 融合の C++ 復活（2026-08-29）

`CUDAFORTRAN_FUSED` を Fortran `2dadc41^` の自然順・長さ 8 内積カーネルとして
C++ に戻した。login GPU 1、`conf_perf_p7.conf` の `DqdtKernel_Type` だけ
`CUDAFORTRAN_FUSED`、3-run 中央値。

| 経路 | Main [ms/step] | µs/stage |
|---|---:|---:|
| `CUDAFORTRAN_FUSED`（CC） | 1.227 | 326.8 |
| `CUDAFORTRAN_FUSED_DFMA` | 1.517 | 424.1 |
| `CUDAFORTRAN_FUSED_TC` | 1.074 | 274.9 |

CC 326.8 µs は旧 Fortran 〜324 µs と同水準。TC は 1.19×（CC 比）、DFMA 比では 1.54×。
点変化係数、`Ne=2³`、`nstep=1` の owned `dqdt` は `FUSED` と `FUSED_TC` がビット一致、
`CUDAFORTRAN_SPLIT` との最大絶対差 1.78e-15。

### 追記 19: p=15…255 CUDA-core 融合の測定（2026-08-29）

同じ CC カーネルを p=15 / 31 / 63 / 127 / 255 に戻した作業ツリー（親 `959ad50`）で、
各次数の既存 `conf_perf_p*` の `DqdtKernel_Type` だけが `CUDAFORTRAN_FUSED` の
3-run 中央値。login GPU 1。FLOP/s は [`README.md`](README.md) まとめ表。

| p | conf | Main [ms/step] | µs/stage | TC / FUSED |
|---:|---|---:|---:|---:|
| 15 | `conf_perf_p15.conf` | 1.583 | 446.7 | 1.64× |
| 31 | `conf_perf_p31_fused.conf` | 3.255 | 998.4 | 2.78× |
| 63 | `conf_perf_p63_fused.conf` | 3.117 | 966.5 | 2.29× |
| 127 | `conf_perf_p127_fused.conf` | 5.033 | 1593.5 | 2.26× |
| 255 | `conf_perf_p255_tc.conf` 相当 | 15.755 | 5252.8 | 5.72× |

点変化係数の owned `dqdt`: p=15 は SPLIT / TC とも 1.78e-15。p=31/63 `Ne=2³`、
p=127 `Ne=1`、p=255 `Ne=1` と `Ne=2` は TC とビット一致。
