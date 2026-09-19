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

（GB200 は本節、H100 / A100 はジョブ投入後に追記。）

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

### 2.4 H100 / A100

（ジョブ投入待ち。`jobs/tsubame_graph_ab_lowp.sh` / `jobs/wisteria_graph_ab_lowp.sh`。
結果は `tsubame/graph_ab_lowp_<id>/`、`wisteria/graph_ab_lowp_<id>/` に戻し、
`graph_ab_lowp_summarize.py` に 3 本まとめて渡す。）

### 2.5 配布 CSV

`../scale-dg-paper/figs/data/graph_ab_lowp.csv`（LF）: 列
`machine,order,path,mma_shape,graph,round,main_per_step_s,steploop_per_stage_s`、
1 run 1 行。GB200 分 144 行（2 次数 × 6 経路 × 2 × 6 ラウンド）。H100
（2 × 8 × 2 × 6 = 192 行）と A100（144 行）が加わると 480 行。原稿側は
§2.2 / §2.3 の表をこの CSV から再計算する。
