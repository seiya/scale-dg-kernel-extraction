# p=7 CUDA-core `FUSED` gap study

GB200 1 GPU、`make CUDA=1 GPUFLAGS=-gpu=cc100`（nvcc `-arch=sm_100`）、
`namelists/perf_p7_fused.conf`（`Ne=32³`、`nstep=20`、graph off）。
親 commit `9dfea48`。対象は `CUDAFORTRAN_FUSED`（`cuda_dg_kernels_fused.cu`）だけ。
`FUSED_TC` / `FUSED_DFMA` は触っていない。

µs/stage = `CUDA device fused tendency` / (19 measured steps × 3 RK)。
採否は占有 GPU 上の 12 回交互 A/B。点変化係数の owned `dqdt`（`Ne=2³`）は
改修前 CC と**ビット一致**、`CUDAFORTRAN_SPLIT` との最大絶対差 1.78e-15。

## 0. ベースライン

login GPU 1、5-run 中央値: Main **1.225 ms/step**、device **326.5 µs/stage**
（`1.86092e-2` s / 57）。占有 GPU（job `70492` の A、c153）は **326.7 µs**。
ncu job `70442`（`--set full`、c398）と `70454`（stall メトリクス）:

| | 値 |
|---|---|
| L1/TEX | **96.0%** |
| Memory throughput | 94.8% |
| DRAM | 39.5% |
| SM | 40.8% |
| 占有率 | 96.6%（レジスタ 32、8 blocks/SM） |
| long scoreboard | 25.2 / 55.2 cycle（**45%**） |
| mio_throttle | 7.3 |
| global ld sectors / request | 95.9 M / 9.31 M = 10.3 |
| セクタ利用 | **23.7 / 32 B** |
| local spill | 524,288 requests（STACK 8） |

1 文: **このカーネルは L1/TEX 上の global ロード待ち（long scoreboard）で律速している。**
未結合の支配因は面 2/4 の stride-8 gather である（TC の `tc_paper_survey` §13 と同じ）。

## 1. 天井（不正アブレーション、login 3-run）

改修前バイナリ `p7cc_base` に対し:

| 消したもの | device s | µs/stage | vs base |
|---|---:|---:|---:|
| ベース | 1.860 | 326.3 | — |
| 内積ループ | 1.457 | 255.6 | **−21.6%** |
| 面 gather 全部 | 1.068 | 187.3 | **−42.6%** |
| 面 2/4 の M 側だけ | 1.736 | 304.6 | **−6.6%** |

改修後（xface + bounds 6 + restrict）では面全体 −38.5%、残る 4 面の M 側 −10.6%、
P 側 −11.6%、内積 −13.1%。P 側は隣接要素なので契約内ではステージできない。
内積は長さ `Nq` の CC スケジュールそのもの。

## 2. 採用

占有 GPU 12 回交互。各行の A は直前の採用形。

| 変更 | job / node | A µs | B µs | device | 機構 |
|---|---|---:|---:|---:|---|
| ±x M を shared へ | `70492` c153 | 326.7 | 324.1 | **−0.79%** | 面 2/4 の stride-8 M gather を体積ロードの値で置換。セクタ 23.7→後段 26.4 / 32。8 blocks のまま STACK 8→16 |
| `__launch_bounds__(256, 6)` | `70500` c384 | 324.7 | 315.1 | **−2.97%** | 40 レジスタで spill 消滅。占有率 75%。L1 待ち行列が 8 CTA より短い |
| `__restrict__` | `70525` c384 | 314.6 | 313.1 | **−0.48%** | エイリアス解消。レンジ非重複（端が 1 ULP） |
| 残る 4 面の M を shared へ | `70531` c384 | 313.1 | 308.5 | **−1.46%** | 面 1/3/5/6 も体積レジスタから書く。shared 21.9→31.1 KB、6 CTA は維持 |
| P 側 `__ldg` | `70532` c384 | 308.7 | 302.8 | **−1.90%** | 隣接 gather を read-only 経路へ。p=31 では +17% だった手が、L1 飽和の p=7 では勝つ |

通算（job `70538` c384、ベース対最終を同一ジョブで交互）:

| | Main [ms/step] | device [s] | µs/stage |
|---|---:|---:|---:|
| 改修前 | 1.223 | 0.018599 | **326.3** |
| 改修後 | 1.152 | 0.017259 | **302.8** |

**−7.20%**（レンジ非重複）。login 5-run 中央値は Main **1.154 ms/step**、device **302.9 µs**。

改修後 ncu（job `70539`）: L1/TEX **96.3%**、Memory 95.3%、DRAM 38.4%、SM 47.6%、
占有率 72.4%、spill **0**、セクタ利用 26.4/32。**律速の種類は変わっていない。**
速くなったのは同じ L1 仕事から M 側 gather と spill を外し、P 側を `__ldg` にしたこと。

µs/stage 302.8 は 4.64 TFLOP/s（ピーク 11.6%）、unique DRAM 5.21 TB/s（65.9%）。
最速は `FUSED_TC`（274.9 µs）のまま。論文の主比は **1.19× → 1.10×**。

## 3. 不採用

| 候補 | 結果 | 理由 |
|---|---|---|
| `__launch_bounds__(256, 4)` | **+21.3%**（job `70512`） | 占有率 50%。L1 待ちは減らず並列度だけ落ちる |
| `(256, 5)` | **+5.75%**（`70524`） | 62.5% でも 6 より遅い |
| `(256, 7)` | **+2.90%**（`70521`） | 32 レジスタのまま STACK 16。7 CTA は spill 付き 8 に近い |
| 面 LDG を内積の前に発行 | **+8.7%**（`70530`） | 40 レジスタ予算を超えて STACK 16。隠したレイテンシより spill が高い |
| `Escale` の `__ldg` | −0.12%、**レンジ重複**（`70535`） | 体積 `Escale` は既に coalesced |

## 4. 残りの天井と終了

契約内で残るのは:

- **P 側 gather（天井 −11.6%）**。隣接要素。速度の事前計算は範囲外。
  オーバーラップは spill で +8.7%。`__ldg` で 1.9% 取ったあとの残り。
- **長さ 8 の shared 内積（天井 −13%）**。CC スケジュールそのもの。MMA 置換は
  `FUSED_TC` の役割。
- 面 store の 2.2-way バンクコンフリクト（ncu 推定 ~22% の shared store wavefront）。
  p=63 §17.4 と同じく L1 global 待ちの横にあり、消しても時間にならない側。

**契約内で実装できる候補の天井は測った。探索を終了する。**
