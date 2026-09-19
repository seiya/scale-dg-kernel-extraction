# CUDA graph と低次の順位 —— nsys 版比較（graph 再生を追えるか）と graph on/off の 3 機インターリーブ計測

> **Status:** `current`（初出 2026-09-19） — **現行ツリーに対する主張**として読んでよい文書。古くなったら更新する。

## 0. 問い

原稿は §timing-protocol で「全計測は `UseCudaGraph = .false.`、理由は
nsys が graph 再生をトレースできなかったため（GB200、nsys 2026.1.1、180 s
timeout、空レポート）」と書き、graph の効果として GB200 の開発ログ 5 組
（`514853f`、非インターリーブ: p=7 `FUSED_TC` −32 µs/step、`FUSED` −34、
p=255 `GEMM_FUSED` −119、`GEMM_CUTE` −132、`FUSED` −59）を引いている。
査読で 2 点:

1. 「Nsight Systems は graph 再生を追えない」は一般には誤り（公式ガイドは
   graph 単位・node 単位のトレースを載せている）。→ **§1**: この環境で
   本当に追えないのかを nsys 5 版 × 次数 × 経路で測り直した。
2. graph 有効時に p ≤ 15 の順位が保たれるかは未計測。→ **§2**: 3 機 ×
   {p=7, 15} × 6 経路 × graph {off, on} を同一ジョブ内で交互に測った。

## 1. nsys は graph 再生を追える —— 止まるのは「2025.6.3 以降の nsys × 本プログラムの p ≤ 15 fused 経路」だけ

### 1.1 結論

- **追える。** graph 粒度（`--cuda-graph-trace=graph`、既定）では
  `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` に graph 起動 1 回 = 1 行が入り、
  1 step のデバイス時間がそのまま読める（p=255 `GEMM_FUSED` 3.012 ms/step、
  p=63 `FUSED_TC` 1.430、p=127 2.000、p=255 `FUSED_TC` 2.808、p=31 1.298、
  p=7 `GEMM_FUSED` 4.544、いずれも nsys 2026.1.1）。node 粒度では再生内の
  カーネルが 1 本ずつ記録される（p=255 `GEMM_FUSED` で 424 カーネル、
  2026.1.1；p=7 `FUSED_TC` で 183 カーネル、2025.1.3）。
- **止まるのは特定の組合せ。** nsys 2025.6.3 / 2026.1.1 / 2026.1.3 / 2026.2.1
  を **p=7 または p=15 の `FUSED` / `FUSED_TC` / `FUSED_DFMA`** に当てると、
  graph の **2 回目の `cudaGraphLaunch`**（nstep=3 の最初の再生）でアプリが
  止まる。nstep=1（capture 無し）と nstep=2（capture + 1 回目の launch）は
  通る。同じ nsys で p=7 `GEMM_FUSED`、p ≥ 31 の `FUSED_TC` は通る。
  **nsys 2025.1.3（cuda/12.9.2 同梱）は p=7 `FUSED_TC` も graph・node 両粒度で
  3〜4 s で完走する。**
- 止まっているのはホスト側: プロセスは CPU 97.8% で回り続け（`ps` の
  STAT `R`）、GPU 使用率は 0%。gdb は `ptrace_scope` で attach できず、
  どの呼び出しで回っているかは未確認（**未解明**。なぜ p ≤ 15 の fused
  経路だけかも問わない）。`Ne` を 8³ / 64³ に変えても、
  `--cuda-event-trace=false` でも同じ。
- 2026-08-25 の job `45707`（2026.1.1、p=7 `FUSED_TC`、180 s timeout、
  `PROCESSED (EMPTY RESULTS)`）はこの再現であり、`overall_summary_report.md`
  §8.3 と `p15` / `p31` / `p1023` の各 gap study の「nsys は graph 経路に
  当てられない」は**この組合せに限る**と読み替える。

### 1.2 測定条件

- 凍結バイナリ `exe_nsys_graph2`（tree `4cc3084`、`make CUDA=1
  GPUFLAGS=-gpu=cc100`）、GB200、driver 580.178.04、`module load nvhpc`
  （nvhpc/26.3）。他版は `module load nvhpc/26.5`、`cuda/12.9.2`、
  `cuda/13.2.2`、`cuda/13.3.1` で切替。
- 入力: `namelists/perf_p<p>_<path>.conf` の `UseCudaGraph` だけを `.true.`
  にしたもの（p=7 は Ne=32³、nstep=20；p=15 は Ne=16³；p=31 は nstep を
  200 → 20；p=127 は nstep=100 のまま）。
- コマンド（毎回 `export DEBUGINFOD_URLS=`）:
  ```
  timeout <T> nsys profile --trace=cuda --sample=none --cpuctxsw=none \
      --resolve-symbols=false --cuda-graph-trace={graph|node} \
      --force-overwrite=true -o <name> ./exe_nsys_graph2 <conf>
  ```
  `T` は 124790 が 300 s、それ以降は 30 s（完走するときは 3〜5 s なので
  十分。2025.1.3 の完走で分かった）。表の rc=124 は timeout。
- 集計は `.nsys-rep` を `nsys export --type sqlite` して
  `CUPTI_ACTIVITY_KIND_KERNEL` / `_GRAPH_TRACE` / `_RUNTIME` の行数を数えた。
  止まった側は `RUNTIME` が 460 行（`cuInit` と `cuGetProcAddress` のみ）、
  `KERNEL` テーブル無し、`CUDA_GRAPH_EVENTS` に "Graph Creation" 1 行 —
  capture までは到達している。
- スクリプト `jobs/nsys_graph2.sh` / `nsys_graph3.sh` / `nsys_graph4.sh`
  （ローカル、未コミット）。出力 `nsys_graph2_124792/` 等（未コミット）。

### 1.3 全試行

| job | node | nsys | 粒度 | conf | rc | 経過 | 記録 |
|---|---|---|---|---|---:|---:|---|
| 124790 | c116 | 2026.1.1 (nvhpc/26.3) | graph | p7 FUSED_TC | 124 | 302 s | 空 |
| 124790 | c116 | 2026.2.1 (nvhpc/26.5) | graph | p7 FUSED_TC | 124 | 302 s | 空 |
| 124790 | c116 | **2025.1.3 (cuda/12.9.2)** | graph | p7 FUSED_TC | **0** | **4 s** | 完走 |
| 124790 | c116 | 2025.6.3 (cuda/13.2.2) | graph | p7 FUSED_TC | 124 | 302 s | 空 |
| 124790 | c116 | 2026.1.3 (cuda/13.3.1) | graph | p7 FUSED_TC | 124 | 302 s | 空（ここで scancel） |
| 124792 | c357 | 2026.1.1 | graph | p7 FUSED_TC | 124 | 31 s | RUNTIME=460、KERNEL 無し |
| 124792 | c357 | 2026.2.1 | graph | p7 FUSED_TC | 124 | 32 s | 同上 |
| 124792 | c357 | **2025.1.3** | graph | p7 FUSED_TC | **0** | 3 s | KERNEL=12、GRAPH_TRACE=19（平均 974.1 µs/step） |
| 124792 | c357 | 2025.6.3 | graph | p7 FUSED_TC | 124 | 31 s | 空 |
| 124792 | c357 | 2026.1.3 | graph | p7 FUSED_TC | 124 | 32 s | 空 |
| 124792 | c357 | 2026.1.1 | node | p7 FUSED_TC | 124 | 31 s | 空 |
| 124792 | c357 | 2026.2.1 | node | p7 FUSED_TC | 124 | 32 s | 空 |
| 124792 | c357 | **2025.1.3** | node | p7 FUSED_TC | **0** | 4 s | KERNEL=183（再生内 60 tendency + 60 rk + 63 halo）、`CUDA_GRAPH_NODE_EVENTS`=28 |
| 124792 | c357 | 2025.6.3 | node | p7 FUSED_TC | 124 | 31 s | 空 |
| 124792 | c357 | 2026.1.3 | node | p7 FUSED_TC | 124 | 31 s | 空 |
| 124792 | c357 | 2026.1.1 | graph | **p255 GEMM_FUSED** | **0** | 4 s | KERNEL=25、GRAPH_TRACE=19（3011.6 µs/step） |
| 124792 | c357 | 2026.2.1 | graph | p255 GEMM_FUSED | **0** | 4 s | 同上 |
| 124792 | c357 | 2026.1.3 | graph | p255 GEMM_FUSED | **0** | 5 s | 同上 |
| 124793 | c116 | 2026.1.1 | graph | p7 **FUSED** | 124 | 32 s | 空 |
| 124793 | c116 | 2026.1.1 | graph | p7 **GEMM_FUSED** | **0** | 4 s | KERNEL=36、GRAPH_TRACE=19（4543.6 µs） |
| 124793 | c116 | 2026.1.1 | graph | p63 FUSED_TC | **0** | 3 s | GRAPH_TRACE=19（1430.0 µs） |
| 124793 | c116 | 2026.1.1 | graph | p127 FUSED_TC (nstep=100) | **0** | 4 s | GRAPH_TRACE=99（2000.2 µs） |
| 124793 | c116 | 2026.1.1 | graph | p255 FUSED_TC | **0** | 3 s | GRAPH_TRACE=19（2807.6 µs） |
| 124793 | c116 | 2026.1.1 | graph | p7 FUSED_TC **nstep=1** | **0** | 4 s | KERNEL=12、graph 無し |
| 124793 | c116 | 2026.1.1 | graph | p7 FUSED_TC **nstep=2** | **0** | 3 s | KERNEL=12、GRAPH_TRACE=1（976.7 µs） |
| 124793 | c116 | 2026.1.1 | graph | p7 FUSED_TC **nstep=3** | 124 | 32 s | 空 |
| 124793 | c116 | 2026.1.1 | graph | p7 FUSED_TC nstep=5 | 124 | 31 s | 空 |
| 124793 | c116 | 2026.1.1 | **node** | p255 GEMM_FUSED | **0** | 4 s | KERNEL=424 |
| 124793 | c116 | 2026.1.1 | graph, `--cuda-event-trace=false` | p7 FUSED_TC | 124 | 32 s | 空 |
| 124793 | c116 | 2026.1.1 | `--trace=cuda-hw` | p7 FUSED_TC | 1 | 1 s | `--trace` 二重指定（スクリプト不備、未試行） |
| 124798 | c051 | 2026.1.1 | graph | **p15 FUSED_TC** | 124 | 32 s | 空 |
| 124798 | c051 | 2026.1.1 | graph | **p31 FUSED_TC** | **0** | 4 s | GRAPH_TRACE=19（1297.5 µs） |
| 124798 | c051 | 2026.1.1 | graph | p7 **FUSED_DFMA** | 124 | 32 s | 空 |
| 124798 | c051 | 2026.1.1 | graph | p7 FUSED_TC Ne=64³ | 124 | 31 s | 空 |
| 124798 | c051 | 2026.1.1 | graph | p7 FUSED_TC Ne=8³ | 124 | 31 s | 空 |
| 124798 | c051 | 2026.1.1 | graph | p7 FUSED_TC（診断） | — | 15 s 時点 | CPU 97.8% (STAT R)、GPU util 0%、gdb attach 不可 (ptrace_scope) |
| 45707 (2026-08-25) | — | 2026.1.1 | graph / node | p7 FUSED_TC | 124 | 180 s | EMPTY RESULTS（当時の記録、`overall_summary_report.md` §8.3） |

まとめると、止まる条件は **{2025.6.3, 2026.1.1, 2026.1.3, 2026.2.1} × {p=7, 15}
× {FUSED, FUSED_TC, FUSED_DFMA} × 2 回目以降の graph launch**。それ以外
（旧版 2025.1.3、p=7 の GEMM_FUSED、p ≥ 31 の fused 経路、GEMM 系全部）は
graph 粒度でも node 粒度でも採れる。Web 上に同じ症状の既知 issue は
見つからなかった（Nsight Systems Release Notes の Known Issues に載る graph
関連 2 件は driver < 12.0 向け；CUPTI 13.x のリリースノートに sm_100 の
graph トレースの既知問題は無い；2026-09-19 調査）。

### 1.4 原稿にどう書くか（案）

> CUDA graphs were disabled for all timings that this paper reports, so that
> the timed configuration launches its kernels in the same way as the
> profiled configuration, whose kernel-resolved analysis (Section X) relies
> on one CUPTI record per kernel launch. Nsight Systems traces graph
> replays in general, at graph or node granularity; in our environment,
> Nsight Systems 2025.6 and later (including the 2026.1.1 that produced all
> profiles here) did not finish on the graph-replay runs of the p ≤ 15 fused
> paths of this program, stalling at the second graph launch with the GPU
> idle, whereas the same binary traced normally at p ≥ 31, on the GEMM
> paths at every order, and under Nsight Systems 2025.1.3 at p = 7 (Section
> Y of the supplemental material). We did not pursue the cause. The graph's
> effect on the rankings is measured directly instead (Section Z): ...

つまり「無効にした理由」は **profiled と timed の起動方法をそろえるため**に
寄せ、nsys の件は「一般の制約ではなく、この環境の特定の組合せで再現する
未解明の停止」として補足に落とす。「Nsight Systems cannot trace a CUDA-graph
replay」に読める文は消す。

## 2. graph on/off の 3 機インターリーブ計測（p=7, 15、6 経路、6 ラウンド）

### 2.1 方法

- `jobs/graph_ab_lowp.sh`（共通本体）＋ `jobs/rikyu_graph_ab_lowp.sh`
  （Slurm）/ `jobs/tsubame_graph_ab_lowp.sh`（SGE）/
  `jobs/wisteria_graph_ab_lowp.sh`（TCS）。`jobs/tsubame_paths.sh` の流儀
  （ルート探索、module、凍結 EXE、metadata.txt）に合わせ、`.gitignore` の
  許可リストに載せて追跡する。集計は `jobs/graph_ab_lowp_summarize.py`。
- namelist: `namelists/perf_p{7,15}_<path>.conf` の `UseCudaGraph` だけを
  `.false.` / `.true.` に置換（Ne、dt、nstep=20、warm-up はそのまま。
  graph モードでは最初の 2 step が warm-up に入る仕様は `main.f90` の通り）。
  H100 では CUTLASS 2 経路を `CutlassMmaShape = "16x8x4"` でも走らせる。
- ループ: `for round in 1..6: for path: for graph in false true`。
- 指標: `Step loop per stage`（原稿が p ≤ 15 の順位に使う軸）と
  `Main per step`。graph モードでは tendency の CUDA event が
  "not measured (graph)" なので device 列は無い。
- 数値検証: 各 (機械, p, 経路) で graph off/on を 1 組走らせ、最終 `q`
  （`SCALE_DG_DUMP_Q`、16,777,216 点）を全点比較。

### 2.2 GB200（Slurm `124807`、node `c051`、2026-09-19）

- tree `ccc26f9`（カーネルは `4cc3084` から不変）、`scale-dg_extraction`
  sha256 `768cead5f00c58e6ec2717092e17b2fed9f4456d9f3d200181501b7b9710c389`
  （`graph_ab_lowp_124807/scale-dg_extraction.gb200`）、driver 580.178.04、
  GB200 2062 MHz、8 分 15 秒、144 run 全て rc=0。
- 数値検証（最終 `q`、16,777,216 点、graph off 対 on）: 12 セル中 10 が
  **ビット一致**、`GEMM_FUSED` の 2 セルだけ max|diff| 7.6e-10（relL2
  p=7 5.7e-10、p=15 1.1e-9）の丸め差。capture 下で cuBLAS が別カーネル
  （集約順の違うもの）を選ぶ分で、AGENTS.md の丸め水準内。

ラウンド中央値 ± MAD（6 ラウンド）、Δ = off − on:

| p | path | stage off [µs] | stage on [µs] | ΔStage [µs] (%) | Main off [µs/step] | Main on | ΔMain [µs/step] (%) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | `FUSED` | 373.43 ± 0.12 | 363.89 ± 0.19 | 9.54 (2.55%) | 1122.1 ± 0.4 | 1093.7 ± 0.6 | 28.4 (2.53%) |
| 7 | `FUSED_DFMA` | 414.71 ± 0.12 | 405.11 ± 0.20 | 9.60 (2.31%) | 1245.9 ± 0.3 | 1217.3 ± 0.6 | 28.6 (2.29%) |
| 7 | `FUSED_TC` | 335.81 ± 0.45 | 326.04 ± 0.43 | 9.77 (2.91%) | 1009.2 ± 1.3 | 980.2 ± 1.3 | 29.0 (2.87%) |
| 7 | `GEMM` | 1671.80 ± 0.40 | 1645.12 ± 0.53 | 26.68 (1.60%) | 5017.2 ± 1.2 | 4937.4 ± 1.5 | 79.7 (1.59%) |
| 7 | `GEMM_CUTE` | 1527.57 ± 0.50 | 1502.60 ± 0.19 | 24.97 (1.63%) | 4584.6 ± 1.4 | 4509.9 ± 0.5 | 74.6 (1.63%) |
| 7 | `GEMM_FUSED` | 1542.47 ± 0.50 | 1515.03 ± 0.22 | 27.44 (1.78%) | 4629.4 ± 1.5 | 4547.3 ± 0.7 | 82.1 (1.77%) |
| 15 | `FUSED` | 373.11 ± 0.38 | 362.78 ± 0.12 | 10.32 (2.77%) | 1121.1 ± 1.1 | 1090.2 ± 0.3 | 30.8 (2.75%) |
| 15 | `FUSED_DFMA` | 526.84 ± 0.38 | 517.18 ± 0.71 | 9.66 (1.83%) | 1582.3 ± 1.1 | 1553.4 ± 2.0 | 28.8 (1.82%) |
| 15 | `FUSED_TC` | 355.29 ± 0.12 | 344.91 ± 0.29 | 10.38 (2.92%) | 1067.6 ± 0.4 | 1036.7 ± 0.9 | 30.9 (2.89%) |
| 15 | `GEMM` | 902.10 ± 0.43 | 885.99 ± 0.24 | 16.11 (1.79%) | 2708.2 ± 1.3 | 2660.1 ± 0.7 | 48.1 (1.78%) |
| 15 | `GEMM_CUTE` | 719.69 ± 0.22 | 701.22 ± 0.20 | 18.46 (2.57%) | 2160.9 ± 0.7 | 2105.7 ± 0.6 | 55.2 (2.55%) |
| 15 | `GEMM_FUSED` | 675.66 ± 0.32 | 652.06 ± 0.16 | 23.60 (3.49%) | 2028.9 ± 0.9 | 1958.3 ± 0.4 | 70.7 (3.48%) |

読み取れること:

- **graph の節約は起動回数で決まり、経路の実行時間とは無関係**: fused 3
  経路（1 stage = 1 tendency kernel）は p=7 / 15 とも **28.4〜30.9 µs/step**
  で一致、GEMM 系（1 stage に 5〜6 kernel）は p=7 で 75〜82、p=15 で
  48〜71 µs/step。stage あたりでは fused 9.5〜10.4 µs、GEMM 系 16〜27 µs。
- **開発ログ 5 組（`514853f`、非インターリーブ）との突合せ**: p=7 `FUSED_TC`
  −32 µs/step に対して今回 −29.0、`FUSED` −34 に対して −28.4。方向・桁は
  一致し、差 3〜6 µs は非インターリーブ計測のノード差・時期差の範囲
  （今回の MAD は 0.3〜1.3 µs）。nsys の launch gap 30.6 µs/step（job
  `45686`、p=7 `FUSED_TC`、graph 無し）とは 29.0 で整合する。p=255 の 3 組は
  本ジョブの対象外（p ≤ 15 が問い）。
- 比率で見ると fused 経路は 2.3〜2.9%、GEMM 系は 1.6〜3.5%。

### 2.3 順位は動かない（GB200）

stage 中央値の比を graph off → on で（`graph_ab_lowp_summarize.py`）:

| p | 比 | off | on | 動き [pt] | 原稿の主張 |
|---:|---|---:|---:|---:|---|
| 7 | `FUSED_DFMA/FUSED_TC` | 1.2350 | 1.2425 | +0.76 | 24 セルの 1 |
| 7 | `FUSED_DFMA/FUSED` | 1.1105 | 1.1133 | +0.27 | 24 セルの 1 |
| 7 | `FUSED/FUSED_TC` | 1.1120 | 1.1161 | +0.40 | 24 セルの 1 |
| 7 | `FUSED_TC/FUSED`（Fig. 1(b)） | 0.8992 | 0.8960 | −0.33 | 図 1(b) の GB200 0.899 |
| 7 | `GEMM_CUTE/GEMM_FUSED` | 0.9903 | 0.9918 | +0.15 | （部分和セル、表外） |
| 7 | `GEMM_CUTE/GEMM` | 0.9137 | 0.9134 | −0.04 | （表外） |
| 7 | `GEMM_FUSED/FUSED_TC` | 4.5934 | 4.6468 | +5.34 | （表外） |
| 7 | `GEMM/FUSED_TC` | 4.9785 | 5.0458 | +6.73 | （表外） |
| 7 | 最速 | `FUSED_TC` | `FUSED_TC` | — | Table fastest-path |
| 15 | `GEMM_FUSED/FUSED_TC` | 1.9017 | 1.8905 | −1.12 | 24 セルの 1 |
| 15 | `FUSED_DFMA/FUSED_TC` | 1.4828 | 1.4995 | +1.66 | 24 セルの 1 |
| 15 | `GEMM_CUTE/GEMM_FUSED` | 1.0652 | 1.0754 | +1.02 | 24 セルの 1 |
| 15 | `FUSED_DFMA/FUSED` | 1.4120 | 1.4256 | +1.36 | 24 セルの 1 |
| 15 | `FUSED/FUSED_TC` | 1.0501 | 1.0518 | +0.17 | 24 セルの 1 |
| 15 | `FUSED_TC/FUSED` | 0.9523 | 0.9507 | −0.15 | Fig. 1(b) の系列 |
| 15 | `GEMM/FUSED_TC` | 2.5391 | 2.5688 | +2.97 | （表外） |
| 15 | `GEMM_CUTE/GEMM` | 0.7978 | 0.7915 | −0.63 | （表外） |
| 15 | 最速 | `FUSED_TC` | `FUSED_TC` | — | Table fastest-path |

**符号が変わったセル: 0**。原稿が GB200 で主張する 8 セル（24 セル中の
GB200 分）＋ Fig. 1(b) ＋ 最速 2 行のうち、比の最大変位は
`FUSED_DFMA/FUSED_TC` p=15 の +1.66 pt、1 に最も近い `GEMM_CUTE/GEMM_FUSED`
p=7 でも 0.9903 → 0.9918 で側は変わらない。動きの符号は一貫していて、
graph は分母（速い側）から分子と同じ µs を引くので **1 より大きい比は
やや大きく、小さい比はやや小さく**なる（節約量は経路によらないため）。
`GEMM_FUSED/FUSED_TC` p=15 だけ逆向き（−1.12 pt）なのは、`GEMM_FUSED` の
節約 23.6 µs/stage が `FUSED_TC` の 10.4 より大きい（起動数の差）ため。

### 2.4 H100（SGE `8714147`、r1n6）と A100（TCS `9659039`、wa34）

- H100: tree `9a7364c`（カーネルは `4cc3084` から不変）、EXE sha256
  `891fe3acb0fa28e55c2f673479e6d5a624b33d876d7dc5b3177158229615e9ad`、driver
  580.105.08、H100 1980 MHz、8 経路（CUTLASS 2 経路は 8x8x4 と 16x8x4）、
  192 run 全 rc=0、11:49〜12:03。**ノード r1n6 は `gg_mig=2` の MIG モード**
  だったが、AGENTS.md の記録どおり stage 時間は正しく（p=7 `FUSED_TC` graph
  off 1049.2 µs は base sweep `8708077` の 1049〜1051 と一致、`FUSED` 1042.2
  も一致）、起動が遅い分は timing 区間の外なのでそのまま採用。
- A100: tree `9a7364c`、EXE sha256
  `b17b424c6b5abe538e6e10a8b04827aee91aeb6b9850bd37192c6400f80580fa`、driver
  535.54.03、A100-SXM4-40GB（applications clock 1095 MHz、max 1410）、6 経路、
  144 run 全 rc=0、11:51〜12:00。p=7 `FUSED_TC` graph off 1677.9 µs は base
  sweep の 1677.7〜1678.2 と一致。
- 数値検証: A100 は 12 セル全てビット一致。H100 は 16 セル中 12 がビット一致、
  `GEMM_FUSED` の 4 セル（両形状）だけ丸め差（p=7 7.6e-10、p=15 8x8x4 1.1e-9 /
  16x8x4 1.3e-8、relL2 ≤ 5.7e-10）。GB200 と同じく capture 下の cuBLAS の
  カーネル選択の差。

ラウンド中央値 ± MAD（6 ラウンド）、Δ = off − on:

| 機 | p | path | shape | stage off [µs] | stage on | ΔStage [µs] (%) | Main off [µs/step] | Main on | ΔMain [µs/step] (%) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| H100 | 7 | `FUSED` | — | 1042.17 ± 0.09 | 1033.77 ± 0.09 | 8.40 (0.81%) | 3129.3 ± 0.2 | 3105.0 ± 0.2 | 24.3 (0.78%) |
| H100 | 7 | `FUSED_DFMA` | — | 1183.95 ± 0.17 | 1175.59 ± 0.21 | 8.37 (0.71%) | 3554.7 ± 0.4 | 3530.4 ± 0.7 | 24.2 (0.68%) |
| H100 | 7 | `FUSED_TC` | — | 1049.24 ± 0.09 | 1040.82 ± 0.22 | 8.42 (0.80%) | 3150.5 ± 0.3 | 3125.7 ± 0.7 | 24.8 (0.79%) |
| H100 | 7 | `GEMM` | — | 2987.51 ± 1.42 | 2971.50 ± 0.35 | 16.01 (0.54%) | 8965.6 ± 4.5 | 8919.0 ± 0.9 | 46.7 (0.52%) |
| H100 | 7 | `GEMM_CUTE` | 8x8x4 | 2887.61 ± 0.23 | 2862.91 ± 0.27 | 24.70 (0.86%) | 8665.9 ± 0.7 | 8592.5 ± 0.9 | 73.4 (0.85%) |
| H100 | 7 | `GEMM_CUTE` | 16x8x4 | 2887.78 ± 0.50 | 2862.89 ± 0.38 | 24.89 (0.86%) | 8666.5 ± 1.6 | 8593.3 ± 1.1 | 73.2 (0.84%) |
| H100 | 7 | `GEMM_FUSED` | 8x8x4 | 2830.81 ± 3.42 | 2781.90 ± 0.89 | 48.91 (1.73%) | 8495.2 ± 10.5 | 8349.8 ± 2.3 | 145.5 (1.71%) |
| H100 | 7 | `GEMM_FUSED` | 16x8x4 | 2828.91 ± 2.38 | 2780.81 ± 0.63 | 48.10 (1.70%) | 8489.8 ± 7.3 | 8347.0 ± 1.4 | 142.8 (1.68%) |
| H100 | 15 | `FUSED` | — | 1014.94 ± 0.47 | 1006.90 ± 0.18 | 8.03 (0.79%) | 3047.5 ± 1.4 | 3024.1 ± 0.5 | 23.3 (0.77%) |
| H100 | 15 | `FUSED_DFMA` | — | 1018.61 ± 0.35 | 1009.38 ± 0.48 | 9.22 (0.91%) | 3058.5 ± 1.0 | 3031.8 ± 1.4 | 26.8 (0.88%) |
| H100 | 15 | `FUSED_TC` | — | 952.53 ± 0.30 | 944.07 ± 0.16 | 8.46 (0.89%) | 2860.2 ± 1.0 | 2835.7 ± 0.6 | 24.5 (0.86%) |
| H100 | 15 | `GEMM` | — | 2098.68 ± 0.18 | 2086.61 ± 0.11 | 12.07 (0.58%) | 6299.0 ± 0.6 | 6263.8 ± 0.5 | 35.1 (0.56%) |
| H100 | 15 | `GEMM_CUTE` | 8x8x4 | 1993.42 ± 0.52 | 1979.58 ± 0.19 | 13.84 (0.69%) | 5983.1 ± 1.6 | 5942.9 ± 0.6 | 40.1 (0.67%) |
| H100 | 15 | `GEMM_CUTE` | 16x8x4 | 1993.76 ± 0.38 | 1978.50 ± 0.23 | 15.26 (0.77%) | 5984.1 ± 1.1 | 5939.8 ± 0.8 | 44.4 (0.74%) |
| H100 | 15 | `GEMM_FUSED` | 8x8x4 | 1889.12 ± 1.67 | 1853.18 ± 0.14 | 35.95 (1.90%) | 5670.2 ± 5.1 | 5563.6 ± 0.6 | 106.7 (1.88%) |
| H100 | 15 | `GEMM_FUSED` | 16x8x4 | 1884.22 ± 0.85 | 1852.59 ± 0.16 | 31.63 (1.68%) | 5655.5 ± 2.5 | 5561.8 ± 0.4 | 93.7 (1.66%) |
| A100 | 7 | `FUSED` | — | 1689.13 ± 0.18 | 1679.41 ± 0.11 | 9.73 (0.58%) | 5069.3 ± 0.5 | 5040.3 ± 0.3 | 29.0 (0.57%) |
| A100 | 7 | `FUSED_DFMA` | — | 1687.74 ± 0.10 | 1678.01 ± 0.16 | 9.73 (0.58%) | 5065.1 ± 0.3 | 5036.2 ± 0.4 | 28.9 (0.57%) |
| A100 | 7 | `FUSED_TC` | — | 1677.88 ± 0.09 | 1668.03 ± 0.23 | 9.86 (0.59%) | 5035.6 ± 0.3 | 5006.1 ± 0.7 | 29.5 (0.59%) |
| A100 | 7 | `GEMM` | — | 5627.73 ± 0.38 | 5601.48 ± 0.24 | 26.25 (0.47%) | 16885.3 ± 1.1 | 16806.8 ± 0.7 | 78.6 (0.47%) |
| A100 | 7 | `GEMM_CUTE` | — | 5337.19 ± 0.06 | 5307.56 ± 0.75 | 29.62 (0.55%) | 16013.6 ± 0.2 | 15925.2 ± 2.1 | 88.4 (0.55%) |
| A100 | 7 | `GEMM_FUSED` | — | 5180.64 ± 0.47 | 5149.42 ± 0.45 | 31.21 (0.60%) | 15544.2 ± 1.3 | 15450.8 ± 1.3 | 93.5 (0.60%) |
| A100 | 15 | `FUSED` | — | 1617.69 ± 0.18 | 1607.12 ± 0.60 | 10.57 (0.65%) | 4855.0 ± 0.5 | 4823.3 ± 1.8 | 31.7 (0.65%) |
| A100 | 15 | `FUSED_DFMA` | — | 1787.32 ± 0.46 | 1776.51 ± 0.21 | 10.81 (0.60%) | 5363.8 ± 1.4 | 5331.6 ± 0.7 | 32.2 (0.60%) |
| A100 | 15 | `FUSED_TC` | — | 1598.35 ± 1.00 | 1588.65 ± 0.38 | 9.69 (0.61%) | 4796.9 ± 3.0 | 4768.1 ± 1.1 | 28.8 (0.60%) |
| A100 | 15 | `GEMM` | — | 3616.07 ± 0.55 | 3595.44 ± 0.39 | 20.62 (0.57%) | 10850.5 ± 1.6 | 10788.6 ± 1.1 | 61.9 (0.57%) |
| A100 | 15 | `GEMM_CUTE` | — | 3231.27 ± 0.37 | 3212.53 ± 0.26 | 18.74 (0.58%) | 9695.9 ± 1.1 | 9639.9 ± 0.8 | 56.1 (0.58%) |
| A100 | 15 | `GEMM_FUSED` | — | 3118.45 ± 1.15 | 3050.39 ± 0.48 | 68.05 (2.18%) | 9357.4 ± 3.4 | 9153.5 ± 1.5 | 203.9 (2.18%) |

順位（stage 中央値の比、graph off → on）:

| 機 | p | 比 | off | on | 動き [pt] | 原稿 |
|---|---:|---|---:|---:|---:|---|
| H100 | 7 | `FUSED_DFMA/FUSED_TC` | 1.1284 | 1.1295 | +0.11 | 24 セル |
| H100 | 7 | `FUSED_DFMA/FUSED` | 1.1360 | 1.1372 | +0.11 | 24 セル |
| H100 | 7 | `FUSED/FUSED_TC` | 0.9933 | 0.9932 | −0.00 | 24 セル |
| H100 | 7 | `FUSED_TC/FUSED` | **1.0068** | **1.0068** | +0.00 | Fig. 1(b) の 1.007 |
| H100 | 7 | `GEMM_CUTE/GEMM_FUSED` 8x8x4 / 16x8x4 | 1.0201 / 1.0208 | 1.0291 / 1.0295 | +0.91 / +0.87 | （表外） |
| H100 | 7 | `GEMM_CUTE/GEMM` | 0.9666 | 0.9635 | −0.31 | （表外） |
| H100 | 7 | `GEMM_FUSED/FUSED_TC` 8x8x4 / 16x8x4 | 2.6980 / 2.6962 | 2.6728 / 2.6718 | −2.52 / −2.44 | （表外） |
| H100 | 7 | `GEMM/FUSED_TC` | 2.8473 | 2.8550 | +0.77 | （表外） |
| H100 | 7 | 最速 | **`FUSED`** | **`FUSED`** | — | Table fastest-path（実測最速 `FUSED`、差 1.0%） |
| H100 | 15 | `GEMM_FUSED/FUSED_TC` 8x8x4 / 16x8x4 | 1.9833 / 1.9781 | 1.9630 / 1.9623 | −2.03 / −1.58 | 24 セル |
| H100 | 15 | `FUSED_DFMA/FUSED_TC` | 1.0694 | 1.0692 | −0.02 | 24 セル |
| H100 | 15 | `GEMM_CUTE/GEMM_FUSED` 8x8x4 / 16x8x4 | 1.0552 / 1.0581 | 1.0682 / 1.0680 | +1.30 / +0.98 | 24 セル |
| H100 | 15 | `FUSED_DFMA/FUSED` | 1.0036 | 1.0025 | −0.11 | 24 セル |
| H100 | 15 | `FUSED/FUSED_TC` | 1.0655 | 1.0666 | +0.10 | 24 セル |
| H100 | 15 | `FUSED_TC/FUSED` | 0.9385 | 0.9376 | −0.09 | Fig. 1(b) 系列 |
| H100 | 15 | `GEMM/FUSED_TC` | 2.2033 | 2.2102 | +0.70 | （表外） |
| H100 | 15 | `GEMM_CUTE/GEMM` | 0.9500 | 0.9482 | −0.18 | （表外） |
| H100 | 15 | 最速 | `FUSED_TC` | `FUSED_TC` | — | Table fastest-path |
| A100 | 7 | `FUSED_DFMA/FUSED_TC` | 1.0059 | 1.0060 | +0.01 | 24 セル |
| A100 | 7 | `FUSED_DFMA/FUSED` | **0.9992** | **0.9992** | −0.00 | 24 セル（1 に最も近いセル） |
| A100 | 7 | `FUSED/FUSED_TC` | 1.0067 | 1.0068 | +0.01 | 24 セル |
| A100 | 7 | `FUSED_TC/FUSED` | 0.9933 | 0.9932 | −0.01 | Fig. 1(b) 系列 |
| A100 | 7 | `GEMM_CUTE/GEMM_FUSED` | 1.0302 | 1.0307 | +0.05 | （表外） |
| A100 | 7 | `GEMM_CUTE/GEMM` | 0.9484 | 0.9475 | −0.08 | （表外） |
| A100 | 7 | `GEMM_FUSED/FUSED_TC` | 3.0876 | 3.0871 | −0.05 | （表外） |
| A100 | 7 | `GEMM/FUSED_TC` | 3.3541 | 3.3581 | +0.41 | （表外） |
| A100 | 7 | 最速 | `FUSED_TC` | `FUSED_TC` | — | Table fastest-path（実測最速 `FUSED_TC`、`FUSED` との差 0.5%） |
| A100 | 15 | `GEMM_FUSED/FUSED_TC` | 1.9510 | 1.9201 | −3.09 | 24 セル |
| A100 | 15 | `FUSED_DFMA/FUSED_TC` | 1.1182 | 1.1183 | +0.00 | 24 セル |
| A100 | 15 | `GEMM_CUTE/GEMM_FUSED` | 1.0362 | 1.0532 | +1.70 | 24 セル |
| A100 | 15 | `FUSED_DFMA/FUSED` | 1.1049 | 1.1054 | +0.05 | 24 セル |
| A100 | 15 | `FUSED/FUSED_TC` | 1.0121 | 1.0116 | −0.05 | 24 セル |
| A100 | 15 | `FUSED_TC/FUSED` | 0.9880 | 0.9885 | +0.05 | Fig. 1(b) 系列 |
| A100 | 15 | `GEMM/FUSED_TC` | 2.2624 | 2.2632 | +0.08 | （表外） |
| A100 | 15 | `GEMM_CUTE/GEMM` | 0.8936 | 0.8935 | −0.01 | （表外） |
| A100 | 15 | 最速 | `FUSED_TC` | `FUSED_TC` | — | Table fastest-path |

### 2.5 3 機まとめ —— 符号が変わったセルは 0

- **24 セル（Table low-order-rankings）、Fig. 1(b) の `FUSED_TC/FUSED`
  （p=7、3 機）、Table fastest-path の p=7 / 15 の 6 行、いずれも graph on で
  符号・最速経路が変わらない。** 24 セル中の最大変位は A100 p=15
  `GEMM_CUTE/GEMM_FUSED` の +1.70 pt（1.036 → 1.053）、次いで GB200 p=15
  `FUSED_DFMA/FUSED_TC` +1.66 pt、H100 p=15 `GEMM_CUTE/GEMM_FUSED` +1.30 pt。
  1 に最も近い 3 セル（A100 p=7 `FUSED_DFMA/FUSED` 0.9992、H100 p=7
  `FUSED/FUSED_TC` 0.9933、A100 p=7 `FUSED_DFMA/FUSED_TC` 1.0059）の動きは
  0.00〜0.01 pt。原稿が引く 2 つの接戦（H100 p=7 `FUSED_TC/FUSED` 1.007、
  A100 p=7 の `FUSED` 対 `FUSED_TC` 0.5%）は graph on でもそれぞれ 1.0068、
  0.59% のまま。
- **節約量は機械と起動数で決まり、経路の実行時間とは無関係**: fused 3 経路
  （1 stage = 1 kernel）は stage あたり GB200 9.5〜10.4 µs、H100 8.0〜9.2、
  A100 9.7〜10.8 で、次数にも経路にもよらない。1 step（3 stage + RK + halo）
  では GB200 28〜31 µs、H100 23〜27、A100 29〜32。GEMM 系は起動数が多い分
  大きく、`GEMM_FUSED` が 3 機とも最大（p=15 で GB200 23.6、H100 32〜36、
  A100 68.1 µs/stage）。
- 比で見ると fused 経路の節約は GB200 2.3〜2.9%、H100 0.7〜0.9%、A100
  0.6〜0.7%。GB200 が大きいのはカーネルが速い（分母が小さい）からで、絶対量は
  3 機とも同程度。
- fused 経路どうしの比はほぼ同じ µs を両辺から引くので、1 より大きい比は
  わずかに大きく、小さい比はわずかに小さくなる（0.4 pt 以下）。動きが 1 pt を
  超えるのは片側が GEMM 系（起動数が違う）のときだけで、それらの比は 1 から
  5% 以上離れている。
- H100 の 16x8x4 と 8x8x4 の差は graph on/off 双方で 0.1% 以内（p ≤ 15 では
  形状が効かない、`subpipe_transfer_model.md` §5AD.8a と整合）。

### 2.6 配布 CSV

`../scale-dg-paper/figs/data/graph_ab_lowp.csv`（LF）: 列
`machine,order,path,mma_shape,graph,round,main_per_step_s,steploop_per_stage_s`、
1 run 1 行、**480 行**（GB200 144 = 2 次数 × 6 経路 × 2 × 6 ラウンド、
H100 192 = 2 × 8 × 2 × 6、A100 144）。書き出しは
`python3 jobs/graph_ab_lowp_summarize.py graph_ab_lowp_124807
tsubame/graph_ab_lowp_8714147 wisteria/graph_ab_lowp_9659039 --csv ...`。
原稿側は §2.2〜§2.5 の表をこの CSV から再計算する。
