# p=15 (Nq=16) — p=7 と p=255 の間を同一 DOF で埋める

測定環境: RIKYU NVIDIA GB200 1 GPU、`make CUDA=1 GPUFLAGS=-gpu=cc100`。
コミット: 本レポートを追加したコミット（親は `e8e1885`）。
ncu は Slurm job `54662`（`job_ncu_p15.sh`）。
入力: `conf_perf_p15.conf`（`NeX=NeY=NeZ=16, PolyOrder=15, nstep=20, UseCudaGraph=.false.`）と
`conf_perf_p7.conf` 相当（`NeX=32, PolyOrder=7`、同 `nstep`）。
検証: `input_p15_val.conf` と `NeX=4` 版、いずれも `SCALE_DG_VARYING_COEFF=1`。

## 1. なぜ p=15 なのか

同一 DOF（16.78 M = 256³）を立方一様メッシュで保つ条件は `NeX * Nq = 256`、
すなわち **Nq が 256 の約数**である。これで候補は 2 冪だけに絞られ、
p = 15, 31, 63, 127 の 4 点しかない。FP64 mma が要求する M,N の 8 倍数・K の 4 倍数は
Nq≥8 の 2 冪なら自動的に満たされるので、**DOF 固定の条件のほうが Tensor Core の
条件より強い**。Nq=12 や 24 のような「TC 的に半端な次数」は検討する前に落ちる。

この 4 点のうち p=15 を最初に取った理由:

- 1 要素の `q` は Nq=8 で 4 KB、**Nq=16 で 32 KB**、Nq=32 で 256 KB。
  227 KB/SM を構造的に超えるのは Nq=32 からなので、**要素まるごとを shared に置く
  設計が通用する最後の点**である。
- 面点率 `6/Nq` が 75% → **37.5%** に落ちる。p=7 の律速だった gather と lift の
  比重が半減する。
- 融合時の理想演算強度 `6*Nq/40` FLOP/B が 1.2 → **2.4** になる。
  GB200 のマシンバランス 5.0 FLOP/B の半分弱。
- `operator_data/p15.dat` が既にある（16 + 256 + 24576 = 24848 値、
  `D1D(1,1) = -60 = -15*16/4`）。p=31 以降と違い LGL 演算子の生成が要らない。

## 2. ホスト側は既に Nq 汎用だった

`main.f90` / `mod_mesh.f90` / `Makefile` / `operator_data/` は**変更不要**。
`OPENACC_ASIS`, `OPENACC_SPLIT`, `CUDAFORTRAN_SPLIT`, `CUDAFORTRAN_GEMM` は
コード変更ゼロで p=15 が走る。`CUDAFORTRAN_FUSED` だけが**独立した 3 層**の
ガードで閉じていた: `mod_advect3d_eq.f90` の設定時チェックと実行時分岐、
および `mod_cuda_dg_kernels.cuf` のラッパ。

p=15 の数値は 4 つの独立実装で一致することを先に確認した
（`CUDAFORTRAN_SPLIT` を基準、`Ne=2³`、点ごと変動係数）:

| 実装 | 基準との相対差 |
|---|---|
| `CUDAFORTRAN_GEMM` | 0（ビット一致） |
| `OPENACC_SPLIT` | 1.26e-15 |
| `OPENACC_ASIS` | 1.26e-15 |

## 3. `tendency_fused_p15_kernel`

p=7 の `tendency_fused_p7_kernel` は**一切変更していない**。
32 レジスタ / 8 ブロック/SM に追い込んだ調整（`execution_times.md` 追記 3）を
壊す理由がないため、Nq=16 は別カーネルとして書いた。

### shared memory 戦略

p=7 の構成をそのまま Nq=16 にすると
`sFluxX/Y/Z` 3 × 4096 × 8 B = 96 KB + `sflux_bnd` 12 KB + `sD1D` 2 KB + `sLift` 0.75 KB
= **約 111 KB/ブロック**で、48 KB の静的 shared 上限を超え carveout の opt-in が
必須になる。`tc_paper_survey_2407.09621.md` §13.4 が記録した **+31%** の罠がこれである。

採用したのは **Nq³ のスクラッチバッファ 1 本を方向ごとに使い回す**構成:

- 各スレッドは担当ノードの `q` を**レジスタに保持**（要素あたり 1 回だけロード）
- フェーズ X: `u` をロードして `buf = q*u` → 収縮 → `Escale_x` を掛けて累算
- フェーズ Y / Z: `v` / `w` で同じことを同じ `buf` に対して行う
- 面フェーズ: 同じ `buf` の先頭 `6*Nq²` を面流束として使い回す
- epilogue: lift を足して `dqdt` へ

global ロードは `q,u,v,w` 各 1 回ずつで**理論最小のまま**、shared は
`32768 (buf) + 2048 (sD1D) + 768 (sLift)` = **35584 B** で静的 shared に収まる
（実測 `cuobjdump -res-usage` で `SHARED:35584 LOCAL:0`）。
代償は `syncthreads` が 1 回から 7 回に増えることと、3 方向が直列化すること。

### スレッド割り当てとバンクコンフリクト

1 ブロック 1 要素、**1024 スレッド、4 ノード/スレッド**。
スレッド `tid` はノード `tid, tid+1024, tid+2048, tid+3072` を持つ。
これらは `(i,j)` を共有し `k` が 4 プレーンずつ違う。

この割り当てだと**レーンが `i` 方向に並び、収縮添字 `l` は warp 内で一様**になるので、
**swizzle なしで 3 方向とも conflict free** になる:

- x 収縮 `buf(l + (j-1)*16 + (k-1)*256)` は `i` に依存しない → half-warp 全体が
  同一アドレス → ブロードキャスト
- y 収縮 `buf(i + (l-1)*16 + (k-1)*256)` は `i` のみに依存 → 16 連続 double
- z 収縮 `buf(i + (j-1)*16 + (l-1)*256)` も同様に 16 連続 double

`cuda_dg_kernels_tc.cu` の `sw_xy` / `sw_z` が必要だったのは、TC カーネルでは
**レーンが収縮添字に対応する**（mma のフラグメント配置がそう要求する）ためであり、
CUDA core 版では割り当ての自由度でそれを回避できる。

## 4. 検証

すべて `SCALE_DG_VARYING_COEFF=1`（点ごとに変動する `u,v,w,Escale,normal_fn,Fscale`）で、
完全な `dqdt(:,1:Ne)` を比較した。

| ケース | 基準 | 相対差 |
|---|---|---|
| `Ne=2³` | `CUDAFORTRAN_SPLIT` | 2.10e-16 |
| `Ne=4³` | `CUDAFORTRAN_SPLIT` | 4.16e-16 |
| `Ne=4³` | `OPENACC_ASIS` | 1.37e-14 |
| `Ne=2³` 定速度 | `CUDAFORTRAN_SPLIT` | 2.50e-16 |

`OPENACC_ASIS` との差が 2 桁大きいのは総和順序の違いによるもので、
他の実装間の差と同じ性質である。CUDA ビルドと非 CUDA ビルドの両方が通ることも確認した。

## 5. 実行時間（`nstep=20`、16.78 M DOF、graph off）

| 実装 | p=7 (`Ne=32³`) Main | p=15 (`Ne=16³`) Main |
|---|---|---|
| `CUDAFORTRAN_FUSED_TC` | **0.02179** | 未実装 |
| `CUDAFORTRAN_FUSED` | 0.02518 | **0.03143** |
| `CUDAFORTRAN_SPLIT` | 0.05214 | 0.10978 † |
| `OPENACC_SPLIT` | 0.05632 | 0.06742 |
| `CUDAFORTRAN_GEMM` | 0.12638 | 0.11186 |

† **`CUDAFORTRAN_SPLIT` の 2 列は同じ実装ではない**。`Nq == 8` のときだけ
`volume_deriv_p7_kernel` に分岐し、それ以外は汎用の `volume_deriv_kernel` が走る
（`mod_cuda_dg_kernels.cuf:1053,1065`）。p=15 の 0.110 秒はこの汎用カーネルの数字で、
p=7 の 0.052 秒と直接は比べられない。**同一実装で p を振れているのは
`OPENACC_SPLIT` と `CUDAFORTRAN_GEMM` の 2 行だけ**である。

融合カーネルの device 時間:

| | device 時間 (s) | 1 stage あたり |
|---|---|---|
| p=7 `FUSED_TC` | 0.016483 | 274.7 µs |
| p=7 `FUSED` | 0.019859 | 331.0 µs |
| p=15 `FUSED` | 0.026035 | 433.9 µs |

**p=15 は p=7 の 1.31 倍の時間で 2 倍の体積演算をこなしている。**
p=15 における既存最速（`OPENACC_SPLIT` 0.0674）に対しては **2.1 倍**、
汎用 `CUDAFORTRAN_SPLIT` に対しては 3.5 倍速い。

## 6. 理論仕事量と達成効率

理論値（1 RK stage あたり、`Nq³*Ne = 16.78 M` DOF）。
体積は 1 ノードあたり `6*Nq + 20` FLOP（3 方向の収縮 `3*2*Nq`、流束積 3、
`Escale` 5、lift 11、最終和 1）、面は 1 面点あたり 21 FLOP。

| | p=7 | p=15 |
|---|---|---|
| 体積 FLOP/ノード | 68 | 116 |
| 面点/要素 | 384 | 1536 |
| 理論 FLOP / call | 1.405e9 | 2.078e9 |
| 理論 DRAM バイト / call | 1.577e9 | 1.326e9 |
| **ncu 実測 DRAM バイト / call** | **1.598e9** | **1.361e9** |

実測が理論の 1.01〜1.03 倍で、**面 gather はほぼ全てキャッシュで吸収されている**。
DRAM トラフィックに無駄はない。

達成効率（ncu ではなく通常実行の device 時間を使用。GB200 のピークは
FP64 CUDA core = FP64 TC = 40.1 TFLOP/s、DRAM 約 7.9 TB/s）:

| | TFLOP/s | ピーク比 | TB/s | ピーク比 |
|---|---|---|---|---|
| p=7 `FUSED_TC` | 5.14 | 12.8% | 5.84 | 74% |
| p=7 `FUSED` | 4.24 | 10.6% | 4.83 | 61% |
| p=15 `FUSED` | 4.79 | 11.9% | 3.14 | **40%** |

**演算強度が倍になった分、p=15 は帯域の壁からは離れた（74% → 40%）。
しかしそれを FLOP レートに変換できておらず、ピーク比は 12% 前後に張り付いたままである。**

## 7. ncu による律速の同定（Slurm job `54662`）

ncu はクロックを固定するので絶対時間は通常実行より 1.3 倍長い。比率で読むこと。

| メトリクス | p=7 `FUSED` | p=15 `FUSED` |
|---|---|---|
| Duration | 510.4 µs | 679.4 µs |
| Memory Throughput（メモリ系の最大） | **95.6%** | **74.1%** |
| DRAM Throughput | 39.5% | 25.3% |
| L2 Throughput | 31.2% | 19.1% |
| Compute (SM) Throughput | 41.4% | 36.0% |
| Executed IPC | 1.20 | **0.89** |
| Registers / thread | 32 | **62** |
| Block Limit Registers | 8 | **1** |
| Block Limit Shared Mem | 9 | **1** |
| Block Limit Warps | 8 | 2 |
| Achieved Occupancy | **96.6%** | **49.2%** |

**p=7 と p=15 で律速の性格が変わっている。**
p=7 はメモリ系 95.6% で飽和した「詰まっている」状態だが、
p=15 は**どのユニットも飽和していない**（メモリ 74%、SM 36%）。
実体は **62 レジスタによる 1 ブロック/SM = 占有率 49%** で、
7 回の `syncthreads` のたびに切り替える先の他ブロックが SM 上に存在しない。
IPC が 1.20 から 0.89 に落ちているのがその帰結である。

## 8. 不採用にした 2 つの調整

**`launch_bounds(1024,2)`**: レジスタは 62 → **32、spill ゼロ**で通った
（`cuobjdump` で `LOCAL:0` を確認）。にもかかわらず device
26.03 → **27.81 ms（+6.8%）**。`Block Limit Shared Mem` が 1 のままなので
ブロック数は増えず、レジスタを削った分だけアドレス再計算が増えただけだった。

**さらに carveout 50% を要求**: device **28.18 ms（+8.2%）**。
`tc_paper_survey_2407.09621.md` §13.4 と同じ結果である。
占有率を上げるにはレジスタと shared の**両方**の制限を同時に外す必要があり、
このカーネルの生きた状態（4 累算器 + 4 個の `q` = 16 レジスタ分）を考えると
32 レジスタで 2 ブロックを成立させる余地は無い。両方とも差し戻した。

## 9. 結論と次の点

- 同一 DOF での 3 点目が入った。`reports/README.md` の
  「p=7 は帯域・L1 律速、p=255 は演算律速」という対比に対し、
  **p=15 はそのどちらでもなく、占有率律速という第三の状態**にある。
- 演算強度を上げれば帯域の壁からは離れられるが、
  **CUDA core の融合カーネルはそれを演算性能に変換できない**。
  1 スレッドが持つ状態量が `Nq` とともに増え、レジスタが先に尽きるためである。
- **Tensor Core 版（Phase C）を試す価値はある**、というのがこの測定の答えである。
  mma は shared からのオペランド読み出しを 8x8 タイル全体で償却し、
  1 レーンあたりの生存レジスタを減らす方向に働くので、
  ここで詰まっている 2 つの要因の両方に直接効く。
  当初の事前予測（「p=15 はまだ帯域側にいるはず」）は**外れた**。
- Nq=32 以降は要素が shared に載らないので、この設計は p=15 で打ち止めである。
  p=31 は別の構造（タイル化）を要する。
