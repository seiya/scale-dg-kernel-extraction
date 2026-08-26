# TMA（Tensor Memory Accelerator）適用調査

作成日: 2026-08-26
対象リポジトリ: `scale-dg-kernel-extraction`
ブランチ / HEAD: `feature/cuda` / `c353993`
対象GPU: NVIDIA GB200 (RIKYU) 1 GPU、`nvcc 13.1`、CUTLASS 4.7.0
Slurm job: `48957`（A-0 `--set full`）、`48976`（B-0 `--set full`）、
`48981`（B-0 マイクロベンチ）、`49009`（A-0 stall 内訳）
測定用実行ファイル: `scale-dg_extraction.tma0`（`c353993` の凍結コピー）

`overall_summary_report.md` §12.9 は 2026-08-25 に TMA を**机上で**調査し、
「FP64 に既製品は無い」「x/y GEMM は天井が上がらない」「狙えるとすれば
z GEMM のレジスタ圧」と書いて実測せずに終わっていた。本レポートはその続きで、
候補を安い順に実測した記録である。

**結論: 採用ゼロ。** ただし 3 つの候補それぞれについて、
「試したら遅かった」ではなく**構造的な理由**が確定した。

| 候補 | 判定 | 決め手 |
|---|---|---|
| A: p=255 z GEMM の epilogue オペランドを TMA 化 | **不採用** | TMA が隠せる long scoreboard stall は全体の **11.3%** しかない。occupancy は shared memory で 4 ブロック/SM に張り付いており、レジスタを返しても上がらない（§3） |
| B: p=7 `FUSED_TC` の volume ロードを TMA 化 | **不採用** | TMA が 8 B 要素で表現できる **4 通りのレイアウト全部**で x/y 収縮が 2-way バンクコンフリクトになる。`sw_xy` が要求する置換を TMA の swizzle 語彙が表現できない（§4） |
| C: z GEMM の mainloop を TMA+DMMA で手書き | **着手せず** | A の測定で前提が消えた。§5 に条件を残す |

---

## 1. 前提条件の実測（Phase 0）

### 1.1 ビルドフラグ

CUTLASS の `cute/arch/config.hpp:41-42` は `CUTE_ARCH_TMA_SM90_ENABLED` を
`CUTLASS_ARCH_MMA_SM90_ENABLED` からしか定義せず、sm_100 では
`cutlass/arch/config.h:91-101` の `__CUDA_ARCH_FEAT_SM100_ALL` か
`CUDA_ARCH_FAMILY(1000)` が要る。実際に `#pragma message` で確認した:

| `nvcc -arch=` | `CUTE_ARCH_TMA_SM90_ENABLED` |
|---|---|
| `native`（= sm_100、現行 `Makefile:11`） | **未定義** |
| `sm_100a` | 定義される |
| `sm_100f` | 定義される |

つまり**現行のビルドフラグでは CuTe の TMA は死んでいる**。TMA を使うなら
その translation unit だけ `-arch=sm_100a` にする必要がある。

### 1.2 `cuTensorMapEncodeTiled` が FP64 で受理する形

`double` の tensor map は作れる（`copy_sm90_desc.hpp:220`）。受理条件を
実機の driver に総当たりで聞いた結果:

| 条件 | 結果 |
|---|---|
| box の各次元 | **256 要素以下**（257, 512 は `CUDA_ERROR_INVALID_VALUE`）|
| global ベースアドレス | **16 B 境界必須**（8 B 境界の `base+1` は拒否、`base+2` は OK）|
| `globalStrides` | **16 B の倍数必須**（lda=255 の 2040 B は拒否、2048 B は OK）|
| swizzle 32B | 最内 box 次元が**ちょうど 4 doubles** のときだけ OK |
| swizzle 64B | 同 **8 doubles** |
| swizzle 128B | 同 **16 doubles** |
| swizzle なし | 最内 box 次元は任意（≤256）|
| 3D map | OK。`escale` の 3 方向断面は box `(64,32,3)` の**記述子 1 本**で足りる |

§12.9 が「未確認」としていた「8 B 要素で 128B swizzle atom がそのまま噛むか」は
**噛む。ただし最内 box 次元 16 に固定される**、が答えである。

落とし穴を 1 つ記録しておく。`globalStrides` は **rank-1 個**の配列で、
dim1 以降の stride しか渡さない（dim0 の stride は要素サイズで暗黙）。
rank 個渡すと dim1 の stride が 0 と解釈され、**エラーにならないまま
第 2 座標が無視される**。

### 1.3 転送そのものの正しさ

値を検証した。列優先 `(512, 64)` の 2D map、box `(64,32)`、swizzle なしで、
タイル座標 6 通り（`(0,0)`,`(0,1)`,`(0,32)`,`(64,0)`,`(64,32)`,`(128,16)`）
すべてで **2048 要素の不一致 0**。swizzle なしのとき smem 上のタイルは
global の box をそのまま列優先で並べたものである。

### 1.4 帯域: TMA に速さの上積みは無い

本リポジトリが実際に必要とする 2 形状で、TMA と素のロードを比べた
（login node、20 回平均）:

| 形状 | 方式 | 時間 | 実効帯域 |
|---|---|---:|---:|
| B: p=7 volume（32768 ブロック × 256 スレッド、4 × 512 doubles、537 MB）| 素の global→smem | 73.22 µs | 7.33 TB/s |
| | **TMA** | **73.15 µs** | **7.34 TB/s** |
| A: z GEMM epilogue（8192 ブロック × 64 スレッド、5 × 64×32、671 MB）| 素の global→レジスタ | 92.24 µs | 7.28 TB/s |
| | **TMA** | **90.61 µs** | **7.41 TB/s** |
| | （参考）素の global→smem、80 KB smem | 669.39 µs | 1.00 TB/s |

**DRAM が飽和している限り TMA に帯域の上積みは無い。** 両形状とも素のロードと
1.8% 以内で並ぶ。3 行目の 7.4× は TMA の勝ちではなく、80 KB の smem を確保して
occupancy を 2 ブロック/SM に落とした私のベースラインの作り方の問題である。
**ベースラインの取り方ひとつで TMA の見かけの効果が 7 倍変わる**ので、
以降はこの罠を避けている。

### 1.5 帯域以外では、TMA は確かに効く

同じ 2 カーネル（同一バイト・同一 shared レイアウト・違いは copy の
発行方法だけ）を ncu で採ると、**帯域が同じでも中身は別物**である
（job `48981`）:

| | `b_plain` | `b_tma` | 比 |
|---|---:|---:|---:|
| duration | 83.65 µs | 74.72 µs | 0.89 |
| **L1/TEX throughput** | 75.62% | **34.52%** | 0.46 |
| **L1 LSU wavefronts** | 10,300,371 | **1,081,842** | **0.11** |
| **global load sectors (L1TEX)** | 16,777,216 | **0** | **0** |
| **shared store バンクコンフリクト** | 1,285,258 | **0** | **0** |
| レジスタ/スレッド | 30 | **16** | 0.53 |
| DRAM bytes | 540,321,536 | 540,384,512 | 1.000 |

| | `a_regs` | `a_tma` | 比 |
|---|---:|---:|---:|
| L1 LSU wavefronts | 5,276,047 | **141,040** | 0.027 |
| global load sectors | 20,971,520 | **0** | 0 |
| 発行命令数 | 23,183,360 | **1,277,960** | 0.055 |
| レジスタ/スレッド | 30 | 16 | 0.53 |

**TMA は「速い転送」ではなく「L1/TEX とレジスタと発行スロットを使わない転送」である。**
したがって効くのは L1・レジスタ・命令発行のいずれかが律速のカーネルに限られる。
以降の 2 つの候補は、まさにその条件を満たすかどうかで判定した。

---

## 2. 候補 A の標的確認: p=255 z GEMM の epilogue（A-0）

`--set full` で `GemmBatchedDqdtAssembly`（融合 epilogue 付き）と
`GemmBatched`（CUTE、素の epilogue）を同一タイル・同一 launch 形状で比較した
（job `48957` / `49009`、p=255 `Ne=1`、`-s 3 -c 1`）。
ncu はクロックを落とすので時間は nsys より長い。

| | FUSED（epilogue 込み） | CUTE（素） |
|---|---:|---:|
| duration | 585.8 µs | 467.1 µs |
| Compute (SM) throughput | 73.25% | 87.20% |
| DRAM throughput | 20.15% | 6.12% |
| L1/TEX throughput | 57.79% | 55.67% |
| L2 hit rate | 49.96% | 72.44% |
| レジスタ/スレッド | 254 | 156 |
| dynamic smem | 49.15 KB | 49.15 KB |
| **Block Limit Registers** | 4 | **6** |
| **Block Limit Shared Mem** | **4** | **4** |
| Theoretical / Achieved occupancy | 12.50 / 12.26% | 12.50 / 12.24% |
| Active warps per scheduler | 1.97 | 1.96 |
| DRAM bytes | 932,820,480 | 231,096,320 |
| 発行命令数 | 136,445,952 | 93,061,120 |
| 整数命令 | 2,060,455,920 | 1,136,656,384 |

stall 内訳（`per_warp_active.pct`、job `49009`）:

| stall 理由 | FUSED | CUTE |
|---|---:|---:|
| **long scoreboard（メモリ待ち）** | **11.33** | 3.27 |
| wait（固定レイテンシ依存） | **36.64** | 35.77 |
| math pipe throttle | 20.44 | 32.43 |
| short scoreboard | 3.12 | 3.95 |
| not selected | 3.69 | 3.67 |
| barrier | 1.53 | 2.44 |
| mio throttle | 0.30 | 1.20 |

### 2.1 判定: 不採用

§12.9 が挙げた狙いは「epilogue に押されているレジスタを mainloop 側から返す」
だった。**その前提が測定で消えた。**

1. **レジスタを返しても occupancy は上がらない。** `Block Limit Shared Mem` が
   両版とも **4** で、CUTE 版はレジスタ 156 本（`Block Limit Registers` = 6）でも
   4 ブロック/SM に張り付いている。律速は shared memory 49.15 KB のほうであり、
   レジスタ 254 → 156 相当まで戻しても theoretical occupancy は 12.5% のまま動かない。
2. **TMA が隠せる待ちが、そもそもほとんど無い。** TMA はメモリレイテンシを
   隠す道具だが、融合版でも long scoreboard は **11.33%** にすぎない。
   支配的なのは固定レイテンシ実行依存 36.6% と演算パイプ 20.4% で、
   どちらもコピーエンジンが触れない。
3. **バイト数は減らない。** 融合 epilogue の増分は DRAM 932.8 − 231.1 =
   **701.7 MB/call** で、これは 5 本の追加オペランド
   （`Dx, Dy, Ex, Ey, Ez` = 5 × 65536×256×8 = **671.1 MB**）とほぼ一致する。
   この読み出しは仕様上必要で、§1.4 の通り TMA でも素のロードでも同じ速度で運ぶ。
4. **shared に載せると悪化する側に効く。** ncu は最上位パイプを
   「Shared（Tensor (DP) が支配、69.7%）」と報告しており、epilogue オペランドを
   smem 経由にすると、すでに最も混んでいる経路に往復を足すことになる。

したがって A は着手しない。

### 2.2 副産物: CUTLASS の FP64 mainloop に 8.5-way の shared store コンフリクトがある

TMA とは無関係だが、A-0 の `--set full` が別の標的を見つけたので記録する。
**z GEMM は融合版・CUTE 版のどちらでも shared store が平均 8.3–8.5-way の
バンクコンフリクトを起こしている。**

| | FUSED | CUTE |
|---|---:|---:|
| shared store requests | 262,144 | 262,144 |
| shared store バンクコンフリクト | 1,172,013 | 1,127,262 |
| 全 shared store wavefronts に占める割合 | **52.78%** | **51.81%** |
| ncu の推定改善余地 | 30.5% | 28.8% |

両版で同じということは、これは自作 epilogue ではなく**CUTLASS 標準の
`MmaMultistage` が `RegularTileAccessIterator` で A/B タイルを書くところ**にある。
z GEMM は現行 p=255 最速パスの単独最大カーネル（nsys 340.0 µs、tendency の 31.6%）
なので、TMA よりこちらのほうが期待値が高い。§5 に次の一手として残す。

---

## 3. 候補 B の標的確認: p=7 `FUSED_TC`（B-0）

`--set full`（job `48976`、p=7 `Ne=32³`、`-s 5 -c 1`、commit `c353993`）:

| | 値 |
|---|---:|
| duration | 403.14 µs |
| Memory throughput | 90.26% |
| **L1/TEX throughput** | **91.91%** |
| DRAM throughput | 49.97% |
| Compute (SM) throughput | 26.86% |
| レジスタ/スレッド | 32 |
| static smem | 16.26 KB |
| Theoretical / Achieved occupancy | 100 / 96.51% |
| Active warps per scheduler | 15.39 / 16 |
| **stall long scoreboard** | **61.9%**（56.4 サイクル中 34.9）|
| global load のセクタ利用率 | 32 B 中 23.7 B |
| shared store バンクコンフリクト | 3.2-way、2,390,739 個（全 store wavefronts の 31.45%）|

**この profile は候補 B にとって理想的である。** L1/TEX が 92% で張り付き、
warp の 62% が L1TEX 待ちで止まっており、§1.5 の通り TMA はその L1 を
一切使わずに global→smem を運ぶ。`tc_paper_survey_2407.09621.md` §12.6 の
「L1 トランザクションを減らす」在庫表に無い 5 番目の案でもあった。

### 3.1 smem 予算

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` で実測した
（`sharedMemPerMultiprocessor` = 233,472 B）:

| static smem / block | blocks/SM |
|---:|---:|
| 16,256（現行） | 8 |
| 20,352（`tc_paper_survey` §9 のレイアウト） | 8 |
| **24,448** | **8** |
| 28,544 | 7 |
| 32,640（`q,u,v,w` を生で 4 本置く） | 6 |

**8 ブロック/SM を保てるステージング枠は +8,192 B**。§12.9 が
「smem 28.2 KB × 8 = 225 KB で使い切っており余地が無い」と書いたのは
`e22dda1` 以前の値で、**現行では余地がある**（この訂正は
`overall_summary_report.md` §12.9 に追記済み）。

### 3.2 判定: 不採用（swizzle が表現できない）

予算は足りる。落ちたのは**レイアウト**である。

このカーネルの速さは 2 つの XOR 置換に依存している（`cuda_dg_kernels_tc.cu:77-90`）:

```
sw_xy(n) = n ^ (((n >> 4) & 1) << 2)     // ノード bit4 をアドレス bit2 へ折る
sw_z (n) = n ^ (((n >> 6) & 3) << 2)     // ノード bit6,7 をアドレス bit2,3 へ折る
```

`tc_paper_survey_2407.09621.md` §10.2 の判定法（16 レーンのフェーズが
コンフリクトフリー ⟺ 動く 4 本のノードビットの像がアドレス下位 4 ビットで
GF(2) 一次独立）を、**TMA が 8 B 要素で符号化できるレイアウト全通り**に
機械的に当てた。置換はハードウェアから読み出したもので、仮定ではない。

| 512 doubles の見せ方 | n0 | n1 | n2 | n3 | **n4** | n5 | x/y 収縮 | z 収縮 |
|---|---|---|---|---|---|---|---|---|
| linear（`(256,2)`、swizzle 無し） | 0x1 | 0x2 | 0x4 | 0x8 | **0x0** | 0x0 | rank 3/4 → **2-way** | rank 2/4 → 4-way |
| `(4,128)` + 32B swizzle | 0x1 | 0x2 | 0x4 | 0x8 | **0x2** | 0x0 | rank 3/4 → **2-way** | rank 2/4 → 4-way |
| `(8,64)` + 64B swizzle | 0x1 | 0x2 | 0x4 | 0x8 | **0x2** | 0x4 | rank 3/4 → **2-way** | rank 2/4 → 4-way |
| `(16,32)` + 128B swizzle | 0x1 | 0x2 | 0x4 | 0x8 | **0x2** | 0x4 | rank 3/4 → **2-way** | rank 3/4 → 2-way |
| （参考）コミット版 `sw_xy` | 0x1 | 0x2 | 0x4 | 0x8 | **0x4** | 0x0 | **rank 4/4 → コンフリクトフリー** | 2/4 |
| （参考）コミット版 `sw_z` | 0x1 | 0x2 | 0x4 | 0x8 | 0x0 | 0x0 | 3/4 | **rank 4/4 → コンフリクトフリー** |

（`§1.2` の受理条件から、8 B 要素で符号化できる swizzle 付きレイアウトは
最内 box 次元 4 / 8 / 16 の 3 通りに限られ、swizzle 無しを足して**この 4 行で全部**である。
表は判定器が linear 行を恒等写像として再現し、コミット版 2 行が §10.2 の
記述どおり 4/4 になることで検証されている。）

原因は 1 行で言える。**TMA の swizzle は行ビットを必ず最下位のチャンクビットから
順に当てる**ので、ノード bit4 はアドレス **bit1** に落ちる。ところが bit1 には
ノード bit1 自身の像がすでにあり、2 本の像が一致して階数が 3 に落ちる。
`sw_xy` が必要とするのは bit4 → **bit2** で、この対応を TMA は表現できない。

収支を見積もると符号は変わらない。TMA で `q,u,v` を smem に落とし、
`q` 倍を mma のオペランドロードに畳み込む版（`tc_paper_survey` §9 版 B と
同じ形で、face gather は global に残す）は 1 スレッドあたり

- global load 8 → 2（−6）、shared store 6 → 2（−4）: 素直に数えて **−20 wavefront/warp**
- x/y 収縮のオペランド読み 4 → 8、かつ**全レイアウトで 2-way**: **+24 wavefront/warp**
- z フラックス用の `q` 読み（linear、コンフリクトフリー）: **+4 wavefront/warp**

で**正味 +8 wavefront/warp**、つまり L1 は減らずに増える。これは
§9（global sectors −26% と引き換えに shared コンフリクト 20 倍、正味 3.7% 遅い）と
§10（コンフリクトフリーな置換は存在するが実機では遅い）が踏んだのと同じ壁で、
今回はその壁が**「TMA の swizzle 語彙にこのカーネルが要る置換が無い」**という
形で現れている。

したがって B も着手しない。**`tc_paper_survey_2407.09621.md` §12.6 の在庫表に
5 行目「TMA で volume を smem へ」を、不採用として追加すべきである。**

---

## 4. 検討して除外した残りの適用先

| 適用先 | 除外理由 |
|---|---|
| p=255 x / y GEMM | FP64 ピークの 86–88%（§12.9）。A-0 で mainloop が Tensor (DP) パイプ 87.2% 律速と確認。GB200 では TC ピーク = CUDA core ピークなので天井が無い |
| `volume_flux_kernel` | 純ストリーミングで smem を使わない。DRAM 83.4% / 7.09 TB/s = ピークの 90%。TMA 化には smem 往復が要り、§1.4 の通り帯域は上積みされない |
| SSP-RK 更新 / halo 更新 | 同上（連続・1 次元・帯域律速） |
| face gather（`VMapM`/`VMapP`） | TMA は矩形タイルしか運べない。不定形 gather は対象外 |
| `cp.async.bulk.prefetch.tensor`（L2 プリフェッチ） | p=7 は 1 ブロック = 1 要素で各要素を 1 回しか読まない。自ブロックの先読みにならず、他ブロックの分を投機するのは前提が脆い |
| TMA multicast（cluster） | 共有されるオペランド（`D1D`）を持つのは x GEMM だが、そこは天井が無い |
| TMA store（S2G） | p=7 epilogue の `dqdt` 書き出しはすでに 16 B coalesce 済み。smem 往復を足すだけ |

---

## 5. §12.9 への訂正と、次に価値のある調査

§12.9 の記述のうち、実測で覆ったもの:

- 「p=7 `FUSED_TC` は smem 28.2 KB × 8 = 225 KB でほぼ使い切っており
  ステージングバッファは occupancy を削る」→ **古い**。`e22dda1` 以降は
  16,256 B で、+8,192 B までは 8 ブロック/SM を保てる（§3.1）。
  ただし B が落ちたのは予算ではなくレイアウトなので、結論は変わらない。
- 「nvfortran は TMA を公開していないので `.cu` への移植が要る」→
  `FUSED_TC` は最初から `.cu` なので当てはまらない（CUDA core 版 `FUSED` には
  当てはまる）。
- 「8 B 要素で 128B swizzle atom がそのまま噛むかは未確認」→ **噛む**。
  ただし最内 box 次元が 16 に固定される（§1.2）。
- 「狙えるとすれば z GEMM のレジスタ圧だけ」→ **その狙いは無い**。
  occupancy は shared memory で 4 ブロック/SM に律速されており、
  レジスタを返しても上がらない（§2.1）。

候補 C（z GEMM の mainloop を CuTe で TMA+DMMA 手書き）は着手していない。
着手を検討するなら、A-0 が示した条件を満たす必要がある:
mainloop 側の律速は Tensor (DP) パイプ 87.2%（CUTE 版）であり、
TMA が減らすのは L1・レジスタ・発行スロットである。したがって
**「mainloop の DMMA 発行率を上げる」以外の効き目は期待できない**。
先に §2.2 の shared store コンフリクトを潰すほうが、同じ場所に対して
ncu 推定 28.8–30.5% と桁が違う。

**次の一手として推奨するのは TMA ではなく §2.2 である。**

---

## 6. 再現手順

Phase 0 の 5 本のマイクロベンチはリポジトリに入れていない（スクラッチ扱い）。
再現に要る情報はすべて本文中にある。実機測定は以下で採った。

```bash
sbatch job_ncu_tma_a0.sh    # A-0  --set full, z GEMM 融合版と CUTE 版
sbatch job_ncu_tma_a0b.sh   # A-0  stall 内訳
sbatch job_ncu_tma_b0.sh    # B-0  --set full, p=7 FUSED_TC
sbatch job_ncu_tma_b0b.sh   # B-0  マイクロベンチの L1 計測
```

ジョブスクリプトは AGENTS.md の規約に従って `module load nvhpc-hpcx`、
`export DEBUGINFOD_URLS=`、`timeout` 付き、凍結実行ファイル
`scale-dg_extraction.tma0` を profile する。`UseCudaGraph = .false.`。
これらのスクリプトと `output/` の生データは commit していない。
