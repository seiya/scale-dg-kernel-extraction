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
| `SCALE_DG_MODULES` | `nvhpc` (RIKYU); `nvhpc/26.1_cuda13.1` when SGE is the scheduler (TSUBAME; there is no `nvhpc/26.1`) | `nvidia/<version>` |
| `SCALE_DG_MODULE_PURGE` | `1` | `0` |
| `SCALE_DG_MACHINE` | `h100` | `a100` |
| `SCALE_DG_PEAK_FP64_CUDA` / `_TENSOR` / `_HBM` | 33.5 / 66.9 / 2.39587 | 9.7 / 19.5 / 1.555 |
| `SCALE_DG_SHAPE_SPECS` | the two `@16x8x4` paths | empty (no sm_90 f64 MMA) |
| `SCALE_DG_RESOURCE` | `gpu_1=1` | the `pjsub` resource line |

The root is found from `SCALE_DG_ROOT`, then `SGE_O_WORKDIR`,
`PJM_O_WORKDIR`, `PWD`, and only then from `$(dirname "$0")/..` as a last
resort -- `$0` is a spool copy under both SGE and TCS, so it must never be the
first answer, and every candidate has to contain `namelists/perf_p7_gemm.conf`
and `Makefile` and be writable before it is accepted.

## Tracked: prediction 3

| file | role |
|---|---|
| `cublas_kernel_names_nsys.sh` | which kernels cuBLAS actually runs for the FP64 volume GEMMs, by name |
| `wisteria_cublas_nsys.sh` | TCS wrapper |

Machine-independent as well; it decides `reports/a100_prediction.md`
prediction 3 and prints the verdict.

## Tracked: the L1 third term

| file | role |
|---|---|
| `l1_roof_ncu.sh` | which roof binds each fused kernel: L1 throughput, FP64 pipe utilisation, shared wavefronts and bank conflicts, occupancy, issue rate |
| `rikyu_l1_roof_ncu.sh` | Slurm wrapper |
| `wisteria_l1_roof_ncu.sh` | Fujitsu TCS wrapper |

Also machine-independent, and tracked for the same reason: the additive
roofline needs a third term whose roof scales with SM count and clock rather
than with FP64 FLOPs, and that can only be fitted with the same measurement
taken on every machine (`reports/a100_report.md` section 10).

`ncu` locks the clocks, so nothing these produce is a wall-time result. They
identify which resource binds; adoption stays on interleaved wall-time A/B.

## Tracked: the SM sub-pipe transfer model

| file | role |
|---|---|
| `subpipe_ncu.sh` | the shared body. Per-kernel SM sub-pipe cycles (`shared` = fp64+tensor, alu, fma, fp64), the per-pipe instruction mix, L1TEX wavefronts / sectors / requests, the L2 hit-miss split, DRAM bytes, occupancy and the twelve issue-stall reasons. Knobs: `ORDERS`, `PATHS`, `SHAPES`, `NSTEP`, `REPEATS`, `CLOCKS`, `LAUNCH_SKIP` / `LAUNCH_COUNT`, `NCU_TIMEOUT`, `TIME_BUDGET`, `OUTDIR`. |
| `rikyu_subpipe_ncu.sh` | Slurm wrapper, p=63/255/511 (GB200 job `92507`) |
| `rikyu_subpipe_shape.sh` | Slurm, `CutlassMmaShape` control at p=511 (job `92628`) |
| `rikyu_subpipe_orders.sh` | Slurm, every order p=7..575 (job `92904`) |
| `rikyu_subpipe_latency.sh` | Slurm, the latency-term counters (job `93312`) |
| `wisteria_subpipe_ncu.sh` | Fujitsu TCS wrapper (TCS `9568133`) |
| `wisteria_subpipe_orders.sh` | Fujitsu TCS, every order (TCS `9569112`) |
| `tsubame_subpipe_ncu.sh` | SGE wrapper (SGE `8621140`) |
| `tsubame_subpipe_shape.sh` | SGE, shape control at p=511 (SGE `8621412`) |
| `tsubame_subpipe_orders.sh` | SGE, every order (SGE `8621935` at 8x8x4, `8621936` at 16x8x4) |

Tracked for the same reason as the two harnesses above, and more sharply:
`reports/subpipe_transfer_model.md` predicts one machine's kernel times, roof
identification and ranking crossovers from another machine's counters, so every
number in it is a ratio between two runs of this body. One shared body on all
three schedulers is what makes that a measurement rather than three
conventions.

`SHAPES` is specific to this harness. On sm_100 and sm_80 every f64 MMA shape
lowers to `DMMA.8x8x4`, so sweeping it is a control that must show nothing; on
sm_90 the wide shapes are single instructions and it must halve the DMMA count
(`reports/sm90_mma_shape_survey.md` section 6, confirmed by counter in
`subpipe_transfer_model.md` sections 2.2 and 5J.1).

`--clock-control none` is the default here, unlike `l1_roof_ncu.sh`: the model
reads utilisations at production clock, and `reports/a100_report.md` section
12.2 shows a locked-clock percentage cannot be lifted to boost. Nothing these
produce is a wall-time result either.

## 追跡スクリプト全一覧（索引漏れの補完）

上の 3 つの表は harness の最初の 3 群しか載せていない。`.gitignore` の
allowlist は 2026-09-09 時点で **`README.md` を除いて 39 本**を追跡しているが、
本文に名前が現れるのはそのうち 9 本だけで、**30 本が索引外**だった。allowlist に
入っているということは、AGENTS.md の言う「公表した機械間比を再現可能にするための
例外」に当たる ＝ 公表済みの数字のどれかの列を支えているということなので、索引漏れ
は `reports/README.md` の索引漏れと同格の欠陥である。以下でそれを埋める。

各行の「何を測るか」は各スクリプトの冒頭コメントから取った。「支えるレポート」は
`grep -l '<script name>' reports/*.md` の実測で、**空欄は「まだどのレポートも
そのファイル名で引用していない」**という意味であり、支えるものが無いという意味では
ない（結果が Slurm/SGE/TCS のジョブ ID で引用されている、あるいは結果待ちの列）。
憶測で埋めていない。

### 共有ボディ（機械非依存。ラッパから `exec` される）

| ファイル | 何を測るか | 支えるレポート |
|---|---|---|
| `tsubame_paths.sh` | path x order の交互実行ベンチ本体。CUDA イベントのデバイスタイマ、round-robin 交互実行、中央値。主軸は "Step loop per stage" | `h100_report.md`, `a100_report.md`, `cublas_emulation_survey.md`, `measurement_inventory.md` |
| `cublas_kernel_names_nsys.sh` | FP64 volume GEMM で cuBLAS が実際に出すカーネル名（`cutlass_80_tensorop_d884gemm` か sm_90 xmma か）。`KINDS` で対象 path を切替（2026-09-08 追加） | `a100_report.md`, `cublas_emulation_survey.md` |
| `l1_roof_ncu.sh` | 加法ルーフラインに足りない第 3 項。カーネルごとに L1 スループット / FP64 パイプ / shared wavefront・bank conflict / occupancy / issue rate のどれが縛るか | `a100_report.md` |
| `roof_bytes_ncu.sh` | 最速 path の**実測** DRAM バイト数とどのルーフが縛るか（tendency 1 回分の per-launch カウンタ、`-k` フィルタ無し） | `measured_dram_and_roofs.md`, `h100_report.md`, `inv_nq_departures.md` |
| `cublas_emulation_ab.sh` | cuBLAS FP64 emulation の A/B（`nat` = native / `emu` = FIXED 55 bit / `adp` = DYNAMIC）。namelist は `perf_p<order>_gemm.conf` を使い emulation ノブだけ差し替える | `cublas_emulation_survey.md` |
| `tsubame_paths_summarize.py` | 上記 paths ログを (order, path) ごとの中央値表に。median と min と "clean" を並べて干渉を検出する | `h100_report.md`, `measurement_inventory.md`, `README.md` |
| `shmem_latency_probe.cu` | **OUT OF ROLE**。DG 演算も mma も含まない shared memory 依存チェーンだけのレイテンシ計測カーネル。実カーネルと同じ grid 32768 / block 128 / static shared 32768 B | `measurement_inventory.md` |

### 群 1: cross-path table（既出。上の「Tracked: the cross-path table」参照）

| スクリプト | 機械 | 何を測るか | 支えるレポート |
|---|---|---|---|
| `rikyu_paths.sh` | GB200 (Slurm) | 共有ボディをそのまま。GB200 の同定情報（machine タグと 3 つのピーク）を上書き（2026-09-09 に追加） | `h100_report.md`, `measurement_inventory.md`, `README.md` |
| `wisteria_paths.sh` | A100 (TCS) | 同上、A100 用。`SCALE_DG_SHAPE_SPECS=` で sm_90 f64 MMA の 2 path を落とす | `a100_prediction.md`, `a100_report.md`, `cublas_emulation_survey.md`, `measurement_inventory.md`, `README.md` |

### 群 2: cuBLAS カーネル名 / prediction 3（既出）

| スクリプト | 機械 | 何を測るか | 支えるレポート |
|---|---|---|---|
| `wisteria_cublas_nsys.sh` | A100 (TCS) | sm_80 が「実家」の機械で cuBLAS が本当に `cutlass_80_tensorop_d884gemm` を出すか＝`a100_prediction.md` 予測 3 の判定 | `a100_report.md` |

### 群 3: L1 第 3 項（既出）

| スクリプト | 機械 | 何を測るか | 支えるレポート |
|---|---|---|---|
| `rikyu_l1_roof_ncu.sh` | GB200 (Slurm) | `l1_roof_ncu.sh` を無改変で。`a100_report.md` 11.7 の取り直し（clock-control 2 モード x 10 交互反復 x 9 構成、`TIME_BUDGET` 90 分） | `a100_report.md` |
| `wisteria_l1_roof_ncu.sh` | A100 (TCS) | 同上、share-short の 2 時間キャップの内側に自分で収まる | `a100_report.md` |

### 群 4: 実測 DRAM バイトとルーフ（`roof_bytes_ncu.sh` のラッパ、索引漏れだった群）

| スクリプト | 機械 | 何を測るか | 支えるレポート |
|---|---|---|---|
| `rikyu_roof_bytes_ncu.sh` | GB200 (Slurm) | `SHAPE=8x8x4` 固定（GB200 では 16x8x4 が無料なので形状は軸でない） | `measured_dram_and_roofs.md` |
| `tsubame_roof_bytes_ncu.sh` | H100 (SGE) | `SHAPE=16x8x4`。`h100_report.md` 8.9.2 の唯一の未解決項目（p=767 が DMMA ルーフ 46.8% / 帯域ルーフ 18.1% でどちらでもない）を閉じる | `measured_dram_and_roofs.md`, `h100_report.md` |
| `wisteria_roof_bytes_ncu.sh` | A100 (TCS) | production クロックでの DRAM トラフィック。A100 は FP64 CUDA core と Tensor Core のルーフ比が 2.058 / 4.134 と離れており、GB200（同値）と H100（1.199 / 1.197）が分離できない 2 つを分離できる | `measured_dram_and_roofs.md` |

### 群 5: CublasEmulation の列（`cublas_emulation_ab.sh` / `cublas_kernel_names_nsys.sh` のラッパ）

`measurement_inventory.md` 19.7 の第 1 項目「`CublasEmulation` は GB200 でしか
測っていない」を閉じる群。GB200 列も**わざと取り直している**:
`cublas_emulation_survey.md` 9.0 の囲み注のとおり、旧ジョブのジョブ内検証がテキスト
ダンプをバイナリとして読んでいたため。

| スクリプト | 機械 | 何を測るか | 支えるレポート |
|---|---|---|---|
| `rikyu_cublas_emu.sh` | GB200 (Slurm) | GB200 列の取り直し ＋ 9.3 第 2 項目の `DYNAMIC` (ADP) を第 3 モードとして交互実行 | （空欄） |
| `tsubame_cublas_emu.sh` | H100 (SGE) | H100 列（未測定だった列）。`gpu_1=1` ＝ H100 1 枚、MIG 断片は使わない | （空欄） |
| `wisteria_cublas_emu.sh` | A100 (TCS) | A100 列。ツールチェーン自体が問いの一部: emulation 呼び出しには `CUBLAS_VERSION >= 130002` が要り、Aquarius 最新の nvidia/25.9 がそれを下回るなら「測れない」が結果 | （空欄） |
| `rikyu_cublas_emu_nsys.sh` | GB200 (Slurm) | 10.2 の未解決問題の GB200 側: emulation は GB200 で**勝ち**（p=511 で 0.947）H100 で**負ける**（2.18）のに native 側は 10.7% しか違わない。差は emulated path の中にあるので、カーネル名で追う | （空欄） |
| `tsubame_cublas_emu_nsys.sh` | H100 (SGE) | 同じ本体・同じ `KINDS` の H100 側。A100 はこの比較に入らない（cuBLAS 12.9.1 は emulation を表現できない、survey 10.3） | `cublas_emulation_survey.md` |

### 群 6: OpenACC の列（`tsubame_paths.sh` / `wisteria_paths.sh` を別ビルドで駆動）

`measurement_inventory.md` 19.7 第 3 項目。OpenACC ビルドは CUDA ビルドと
オブジェクトが非互換なので **worktree で別ビルドし、`EXE` で渡す**。
`VALIDATE=0` は手抜きではなく、ACC ビルドには `CUDAFORTRAN_*` path が無く
ボディの参照カーネルが存在しないため（数値検証は CUDA 側ジョブの仕事）。

| スクリプト | 機械 | 何を測るか | 支えるレポート |
|---|---|---|---|
| `rikyu_acc.sh` | GB200 (Slurm) | GB200 の `OPENACC_ASIS` / `OPENACC_SPLIT` を、H100/A100 と同じ "Step loop per stage" 軸で取り直す（18.8 は別軸だったので 3 機械を割れなかった） | `measurement_inventory.md`, `README.md` |
| `tsubame_acc.sh` | H100 (SGE) | H100 の OpenACC 列。worktree ビルド手順はこのファイルのヘッダが正典 | （空欄） |
| `wisteria_acc.sh` | A100 (TCS) | A100 の OpenACC 列。`EXE` は**絶対パス**必須（job 9539009 は `$PWD` をスケジューラ経由で渡して 24 run 全滅した） | `measurement_inventory.md`, `README.md` |

### 群 7: Ozaki の列とクラッシュ診断

| スクリプト | 機械 | 何を測るか | 支えるレポート |
|---|---|---|---|
| `tsubame_ozaki.sh` | H100 (SGE) | H100 の `GEMM_OZAKI1/2` 列。追跡ボディを `SCALE_DG_PATHS_<order>` で駆動するだけで、測定ロジックはここに無い。高次の OOM は失敗でなく結果（`rc != 0` を印字して継続） | `ozaki2_implementation_report.md` |
| `wisteria_ozaki.sh` | A100 (TCS) | A100 の Ozaki 列。機械差は `wisteria_paths.sh` から継承。40 GiB がボトルネックで p=767/1023 は外す | `ozaki2_implementation_report.md` |
| `tsubame_ozaki_sanitizer.sh` | H100 (SGE) | `ozaki2_implementation_report.md` 8.4 の予測 O1（INT8 バッファのパディング）が**外れた**後の compute-sanitizer 実行。CUDA のエラーは非同期なので "x-deriv" は気づいた場所にすぎず、カーネル名とアドレスが要る | `ozaki2_implementation_report.md` |
| `wisteria_ozaki_sanitizer.sh` | A100 (TCS) | 同じ診断の A100 側。job 9539840 が両構成で exit 127 になったのは module load が無く nvcc が PATH に無かったため（修正済み） | `ozaki2_implementation_report.md` |

### 群 8: p=511 のばらつきと機械間の順位反転

`measurement_inventory.md` 19.7 最終項目と 21.x / 23.x の系列。**ばらつきの機構**
（A100 で同一バイナリの 12 プロセスが 2.3-2.9% 違い、命令数・DRAM バイト・occupancy
が全て同一）と、**順位が機械間で反転する理由**（GB200: GEMM_FUSED 勝ち 0.950 /
A100: FUSED_TC 勝ち 1.065）の 2 本立て。

| スクリプト | 機械 | 何を測るか | 支えるレポート |
|---|---|---|---|
| `wisteria_p511_spread.sh` | A100 (TCS) | 最も安い問いから: ばらつきは**再現するのか**。20 ラウンド交互実行、プロファイラ無し | （空欄） |
| `wisteria_p511_clock.sh` | A100 (TCS) | SM クロックと電力をサンプリングし、各 run の時間とその間のクロックを相関させる。ノイズは GB200 < H100 < A100 で電力ヘッドルームの順 | `measurement_inventory.md` |
| `tsubame_p511_clock.sh` | H100 (SGE) | 同じ実験の H100 側（予測 Q4）。stage MAD は H100 0.103% / A100 0.244% / GB200 0.011% なので「同符号でより小さい効果」が期待値。5 path 版でないと harness の protocol に一致しない | `measurement_inventory.md` |
| `wisteria_p511_dir1_ncu.sh` | A100 (TCS) | 21.13 が絞り込んだ 1 launch（`tendency_p255_kernel` DIR1 だけが 2.84% 動き、DIR0/DIR2 は 0.2%）が 8 プロセスでも残るか、残るとき何が違うか | `measurement_inventory.md` |
| `rikyu_p511_dir1_ncu.sh` | GB200 (Slurm) | その**対照**。GB200 のこのセルの spread は 0.011% なので DIR1 は動かないはず。ついでに pass B の `.avg/.max/.min` ロールアップ名を A100 ジョブの前に検証する（名前が弾かれると ncu ごと落ちるため別 invocation） | （空欄） |
| `wisteria_p511_dir2_stalls.sh` | A100 (TCS) | 21.16 が残した唯一の未測定量: 遅いプロセスでどの stall が伸びるか。stall reason は手書きせず列挙する | （空欄） |
| `wisteria_p511_ctamap_ncu.sh` | A100 (TCS) | 21.19 の残り 1 つ「なぜ待ちが動くのか」。候補 (a) CTA-to-SM 割当順 (b) L1/shared carveout (c) ECC/refresh のうち残ったもの | `measurement_inventory.md` |
| `wisteria_shmem_probe.sh` | A100 (TCS) | **OUT OF ROLE**。23.6 の問い「その shared レイテンシ変動は**このカーネル**の性質か**機械**の性質か」を、カーネルを取り除いて測る。global チェーンが対照 | `measurement_inventory.md` |
| `rikyu_shmem_probe.sh` | GB200 (Slurm) | **OUT OF ROLE**。上の対照。GB200 では効果が存在しない（21.15 で 0.08-0.22%）ので probe は平坦なはず。平坦でなければ probe が自分のノイズを測っており A100 側は読めない | （空欄） |
| `rikyu_p511_reversal_ncu.sh` | GB200 (Slurm) | 順位反転の仮説とメトリック選択の正典。H1 = トラフィック（GEMM_FUSED は 145.3 B/node 対 unique 64.5、TC 系は p=255 で 66 前後 ≒ 2.2 倍）。A100 の HBM は SM サイクルあたり 10.2 B、GB200 は 25.2 B なので同じ超過分が約 2.5 倍高くつくはず | `a100_report.md` |
| `wisteria_p511_reversal_ncu.sh` | A100 (TCS) | 同じ研究の A100 半分。仮説とメトリック選択は rikyu 版のヘッダ | `a100_report.md` |

### 群 9: コンパイラ感度（この campaign が制御できなかった唯一の変数）

| スクリプト | 機械 | 何を測るか | 支えるレポート |
|---|---|---|---|
| `wisteria_compiler_sens.sh` | A100 (TCS) | `a100_report.md` 8.2 が記録する未制御変数: 3 機械が 3 コンパイラ（RIKYU nvhpc 26.3 / TSUBAME 26.1 / Wisteria 25.9）。除去はできないが**上限は測れる**ので、1 機械 1 ジョブで nvidia/23.3 と 25.9（Aquarius で取れる最大の隔たり）を同一次数で走らせ、25.9 対 26.3 の差が最大でいくらかを押さえる。codegen は 1 次の効果である（`h100_report.md` 8.7.3 は SPLIT の反転を nvfortran のレジスタ割当に帰し 13-20% と値付けした） | （空欄） |

## 共有ボディとラッパの関係

追跡分は「1 つの共有ボディ ＋ 機械ごとの薄いラッパ」という構造を 5 群に増やして
いる（README 本文の表は最初の 3 群のままだった）。ラッパは `exec` するだけで、
測定ロジックを持たない。

```
tsubame_paths.sh  (path x order ベンチ本体、機械非依存)
├── rikyu_paths.sh          GB200 / Slurm
├── wisteria_paths.sh       A100 / TCS ──┬── wisteria_acc.sh        (OpenACC 列)
│                                        ├── wisteria_ozaki.sh      (Ozaki 列)
│                                        └── wisteria_p511_spread.sh(20 ラウンド)
├── tsubame_acc.sh          H100 / SGE   (別 worktree の ACC ビルドを EXE で渡す)
├── tsubame_ozaki.sh        H100 / SGE   (SCALE_DG_PATHS_<order> で駆動)
├── rikyu_acc.sh            GB200 / Slurm(同上)
└── wisteria_compiler_sens.sh            (ボディを 2 コンパイラで 2 回呼ぶ)

cublas_kernel_names_nsys.sh  (cuBLAS のカーネル名を nsys で取る)
├── wisteria_cublas_nsys.sh      A100 / TCS   (a100_prediction.md 予測 3)
├── rikyu_cublas_emu_nsys.sh     GB200 / Slurm(KINDS で emulated path へ)
└── tsubame_cublas_emu_nsys.sh   H100 / SGE   (同上)

l1_roof_ncu.sh  (どのルーフが縛るか。ORDERS/PATHS/KERNELS/CLOCKS で汎用化)
├── rikyu_l1_roof_ncu.sh          GB200 / Slurm
├── wisteria_l1_roof_ncu.sh       A100 / TCS
├── rikyu_p511_reversal_ncu.sh    GB200 / Slurm(ORDERS=511, PATHS=TC/GEMM_FUSED/GEMM)
└── wisteria_p511_reversal_ncu.sh A100 / TCS   (同上)

roof_bytes_ncu.sh  (実測 DRAM バイトとルーフ)
├── rikyu_roof_bytes_ncu.sh     GB200 / Slurm (SHAPE=8x8x4)
├── tsubame_roof_bytes_ncu.sh   H100 / SGE    (SHAPE=16x8x4)
└── wisteria_roof_bytes_ncu.sh  A100 / TCS

cublas_emulation_ab.sh  (native vs FIXED vs DYNAMIC の A/B)
├── rikyu_cublas_emu.sh     GB200 / Slurm (取り直し ＋ ADP)
├── tsubame_cublas_emu.sh   H100 / SGE
└── wisteria_cublas_emu.sh  A100 / TCS

shmem_latency_probe.cu  (OUT OF ROLE。shared レイテンシだけのカーネル)
├── wisteria_shmem_probe.sh  A100 / TCS   (本番)
└── rikyu_shmem_probe.sh     GB200 / Slurm(対照。平坦であるべき)

ボディを持たない単独ジョブ（対応する共有ボディが無いもの）:
  tsubame_p511_clock.sh / wisteria_p511_clock.sh      クロックと電力のサンプリング
  rikyu_p511_dir1_ncu.sh / wisteria_p511_dir1_ncu.sh  DIR1 単発の per-process 差
  wisteria_p511_dir2_stalls.sh                        どの stall が伸びるか
  wisteria_p511_ctamap_ncu.sh                         CTA マップ / ページ配置
  tsubame_ozaki_sanitizer.sh / wisteria_ozaki_sanitizer.sh  compute-sanitizer
```

### ラッパが上書きする変数（機械の同定とルーフ）

ボディ側の既定値は**すべて H100** である。TSUBAME がこのボディを最初に走らせた
からで、それ以外のラッパは全部上書きしなければならない。

| 変数 | ボディの既定 (H100) | `rikyu_paths.sh` (GB200) | `wisteria_paths.sh` (A100) |
|---|---|---|---|
| `SCALE_DG_MACHINE` | `h100` | `gb200` | `a100` |
| `SCALE_DG_PEAK_FP64_CUDA` | 33.5 | 40.1 | 9.7 |
| `SCALE_DG_PEAK_FP64_TENSOR` | 66.9 | **40.1**（GB200 は CUDA core と同値。Ampere/Hopper の 2 倍優位は無い） | 19.5 |
| `SCALE_DG_PEAK_HBM` | 2.39587 | 7.9 | 1.555 |
| `SCALE_DG_RESOURCE` | `gpu_1=1` | `sbatch --gpus=1 (GB200 1 GPU)` | `pjsub gpu=1 (A100 40GiB)` |
| `SCALE_DG_MODULES` | `nvhpc` | `nvhpc` | `nvidia/25.9` |
| `SCALE_DG_MODULE_PURGE` | `1` | `1` | `0` |
| `SCALE_DG_SHAPE_SPECS` | `@16x8x4` の 2 path | 既定のまま | 空（sm_80 に f64 MMA の 16x8x4 が無い） |

この 4 つ（`MACHINE` と 3 つの `PEAK_*`）は `metadata.txt` に書き出され、
レポートが「その run はどのピークに対して採点されたのか」を推測せずに済むための
ファイルになっている。したがって上書き漏れは**そのジョブの成果物のラベルが間違う**
という形で効く。

> **2026-09-09 の修正**: `rikyu_paths.sh` はこの 4 つ（＋ `SCALE_DG_RESOURCE`）を
> 上書きしていなかった。`wisteria_paths.sh` は最初から 4 つとも上書きしている。
> このため **job 81583 (2026-09-05、9 次数の取り直し) の `metadata.txt` は
> `machine=h100` と H100 のルーフ (33.5 / 66.9 / 2.39587) を持っている**。
> 測定値そのものは GB200 のもので正しく、ラベルだけが誤りである。過去の表は書き
> 換えず、この注記で対応する（AGENTS.md「測定済みの表は書き換えない」）。
> 2026-09-09 以降の GB200 run は `machine=gb200` と 40.1 / 40.1 / 7.9 を持つ。

同じ上書きは `rikyu_acc.sh`（`gb200` / `nvhpc`）と `wisteria_compiler_sens.sh`
（`a100` ＋ 3 ピーク ＋ `SCALE_DG_MODULE_UNLOAD=nvidia`）も自分で行っている。
`wisteria_ozaki.sh` と `wisteria_p511_spread.sh` と `wisteria_acc.sh` は
`wisteria_paths.sh` を経由するので上書きを継承し、A100 の設定を 1 箇所に保つ。

### ボディ側のノブ（ラッパが測定内容を変えるための入口）

| ボディ | 主なノブ |
|---|---|
| `tsubame_paths.sh` | `ORDERS_A/B/C`, `ROUNDS`/`ROUNDS_HI`/`ROUNDS_XHI`, `VALIDATE`, `TIMING`, `SCALE_DG_PATHS_<order>`（1 次数の path 一覧を差し替える再実行用）, `EXE`, `OUTDIR` |
| `cublas_kernel_names_nsys.sh` | `ORDERS`, `KINDS`（既定 `GEMM GEMM_CUTE`。emulated path を見るラッパはここを変える。**2026-09-08 に追加されたので、それ以前のチェックアウトはこのノブを黙って無視する**）, `NSTEP` |
| `l1_roof_ncu.sh` | `ORDERS`, `PATHS`, `KERNELS`, `CLOCKS`, `REPEATS`, `LAUNCH_SKIP`/`LAUNCH_COUNT`, `NCU_TIMEOUT`, `TIME_BUDGET` |
| `roof_bytes_ncu.sh` | `SHAPE`, `CLOCK`, `NSTEP`, `SKIP_CALLS`/`COUNT`, `REPLAY`, `NCU_TIMEOUT`, `TIME_BUDGET` |
| `cublas_emulation_ab.sh` | `EMU_MODES`（`nat emu adp`）, `ROUNDS`/`ROUNDS_HI`, `VALIDATE`, `TIMING`, `SCALE_DG_MODULE_UNLOAD` |

`REPEATS=1 CLOCKS=base` は `l1_roof_ncu.sh` 系の旧ジョブ（Slurm 82908、TCS 9523261）
を正確に再現する設定である。

`ncu` を使うもの（`*_l1_roof_*`, `*_roof_bytes_*`, `*_p511_*_ncu`, `*_reversal_*`）
はクロックを固定するので、そこから出る数字は wall time の結果ではない。何が縛って
いるかを同定するためのもので、採否は交互 A/B の wall time で決める（AGENTS.md）。

## Not tracked

One-off ablations, sweeps and `ncu` probes (`ab_*`, `sweep_*`, `time_*`,
`ncu_*`, `nsys_*`). Names are `{tool}_{scope}_{topic}.sh`. Durable profiler
commands belong in `reports/`, not here. Output trees written by these jobs
are not tracked either.

その 5 プレフィクスは実態を覆っていない。commit `490e94e`（2026-09-09）時点で
`jobs/` の未追跡エントリは 299 あり、内訳は `ab_*` 63 / `ncu_*` 81（うち 1 つは
出力ディレクトリ `ncu_out`）/ `nsys_*` 27 / `time_*` 18 / `sweep_*` 1 に対し、
**どのプレフィクスにも当てはまらないものが 66**、実験専用の `.conf` が 31 本、
出力ディレクトリが 13 個ある（63+81+27+18+1+66+31+13 − `ncu_out` の重複 1 ＝
299）。当てはまらない 66 は `jobs/MAPPING.md` と `analyze_78846.py` を含み、残る
64 本のスクリプトは
`{machine}_{topic}.sh`（`rikyu_recheck_all.sh` など）、`ax_*` / `hi_*` / `db_*` /
`ws_*` の系列、および `s{N}_{step}.sh`（`s18_build.sh` → `s18_ab.sh` →
`s18_ncu.sh` のような段階名）である。これらは追跡対象ではないので命名を揃える
必要は無いが、「未追跡＝この 5 つ」という読み方はできない。

未追跡だが追跡ファイルから名前で参照されているものが 3 つある（`grep -oE
'jobs/[A-Za-z0-9_]+\.(sh|py|cu)'` を追跡ファイル全部にかけた実測）:

| 参照元（追跡） | 参照先（未追跡だが実在） |
|---|---|
| `tsubame_paths.sh` ヘッダ | `tsubame_ch7.sh`（前身。`reports/h100_report.md` §3 の測定） |
| `tsubame_paths.sh` ヘッダ | `tsubame_mma_shape.sh`（`sm90_mma_shape_survey.md`） |
| `l1_roof_ncu.sh`:138 | `split_deriv_ncu.sh`（job 81637 の教訓の出典） |

加えて `cublas_kernel_names_nsys.sh`:28 のヘッダは投入例として
`sbatch jobs/rikyu_cublas_nsys.sh` を挙げるが、**このファイルは現ツリーにも
`git log --all` にも存在しない**（GB200 側の同じボディを走らせるのは
`rikyu_cublas_emu_nsys.sh`）。ヘッダのこの 1 行は宙に浮いた参照である。

以下は `tsubame_ch7.sh` についての補足:
`tsubame_paths.sh` が置き換えた前身であり、h100_report.md §3 の測定はそれで
取られている。復元元は commit `6466dc0`（`jobs/` へ移す前の、リポジトリ直下
`job_tsubame_ch7.sh`）。**`9eed9e5` ではない** -- あれは §3 が測定したツリーの
commit で、`jobs/` を一切含まない。`6466dc0` 版との差はチェックアウトを探す
24 行の追加だけで、測定内容は同じ（`jobs/MAPPING.md` §1）。

## Tracked: CUDA graph on/off at the production orders

| file | role |
|---|---|
| `graph_ab_lowp.sh` | the shared body. p=7 and 15, the six paper paths, `UseCudaGraph` off and on adjacent inside each round, six rounds; final-`q` agreement off vs on, one pair per cell. Knobs: `ROUNDS`, `ORDERS`, `SCALE_DG_PATHS`, `SCALE_DG_SHAPE_SPECS`, `VALIDATE`, `OUTDIR`, `EXE`. |
| `rikyu_graph_ab_lowp.sh` | Slurm wrapper (GB200 job `124807`) |
| `tsubame_graph_ab_lowp.sh` | SGE wrapper, adds the two CUTLASS paths at 16x8x4 (SGE `8714147`) |
| `wisteria_graph_ab_lowp.sh` | Fujitsu TCS wrapper (TCS `9659039`) |
| `graph_ab_lowp_summarize.py` | round median / MAD, the graph's saving per path, and the sign of every p <= 15 ranking the paper claims, off vs on; `--csv` writes `figs/data/graph_ab_lowp.csv` |

Report: `reports/cuda_graph_lowp.md` section 2.

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
