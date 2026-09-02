# arXiv:2407.09621 からの最適化取り込み候補 調査メモ

作成日: 2026-08-25
対象論文: Cu Cui, *Acceleration of Tensor-Product Operations with Tensor Cores*,
arXiv:2407.09621v1 (Heidelberg Univ., 2024). NVIDIA A100 SXM4, SIPG-DG Laplacian,
matrix-free / sum-factorization, FP64 および FP16 Tensor Core。

対象リポジトリ: `scale-dg-kernel-extraction`
ブランチ / HEAD: `feature/cuda` / `299a868`
対象GPU: NVIDIA GB200 (RIKYU), sm_100

本メモはコミット対象外の調査資料（`AGENTS.md` の方針に従う）。

---

## 0. 出発点となる実測値

`output/` の既存 ncu 測定（p=7, `Ne = 32^3`, GB200）:

| kernel | duration | Compute(SM) | L1/TEX | DRAM | regs | smem/block | achieved occ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tendency_fused_p7_kernel` (CUDA core) | 551 µs | 38.5% | 89.6% | 36.5% | 42 | 15.87 KB | 59.97% |
| `tendency_fused_p7_tc_kernel` (Tensor Core) | 666 µs | **22.6%** | **81.3%** | 30.2% | 32 | 28.16 KB | 97.1% |

Tensor Core 版は演算器が 22% しか回らず、L1 / shared パイプで律速している。
これは論文が "MMA basic"（バンクコンフリクトあり）として報告した状態と症状が一致する
（論文では MMA basic → MMA CF で MIO stall 11.58 → 2.11、性能は 2 倍以上）。

注意: `output/` の従来の ncu 収集は `--set basic` で、
`GPU Speed Of Light Throughput` / `Launch Statistics` / `Occupancy` /
`GPU and Memory Workload Distribution` のみを含み、
Memory Workload Analysis（バンクコンフリクト指標）が未収集だった。
→ **§5 で測定済み（Slurm job 43554, 2026-08-25）**。

---

## 1. 論文側の主張の要点

- 中核は「WMMA → MMA への置き換え」ではなく **MMA + XOR 置換によるコンフリクトフリー
  shared memory レイアウト**。これ単体で 2 倍以上の寄与。
- 最良カーネル `MMA CF` は FP64 で 8 TFLOP/s（A100 FP64 TC ピークの 45%）、
  高度に最適化された CUDA Core 実装に対し 2.3×。
- WMMA API はレイアウト制約が厳しく、3D では z 方向が CUDA Core 落ちして
  演算の 2/3 しか TC 化できない。
- 1 要素 = 1 スレッドブロック、2D スレッド構造、warp が複数スライスを担当。
  全データを一度 shared memory に載せてから計算する。
- 1D 作用素行列の置き場所: constant は ADU 逼迫、texture はレジスタスピル、
  **shared が最良**（論文 Table 2）。
- 内側ループのアンロールは逆効果になりうる（216 → 254 regs でスピル、10% 悪化）。
- shared memory ルーフライン: `B = #SM × #banks × word length × clock`
  （A100 で 17.145 TB/s）、`R = B·F/(d_r + d_w)`。DRAM ルーフラインより実態に合う。
- 任意サイズ行列: スレッドのみ padding は分岐発散で不利。
  **メモリとスレッドの両方を padding** し、3D では x0/x1 方向のみで足りる。
  ただし N=10〜14 は N=16 と同じ演算量になり実効効率が落ちる（論文 Fig.10）。
- FP16 + error correction（Algorithm 1, 2 項分割 `δA = (A - half(A))·2^11`）で
  FGMRES + multigrid 前処理が 4.29×。精度は FP32 相当。

---

## 2. 取り込み候補（優先度順）

### A-1. shared memory のコンフリクトフリー化【最優先】

本リポジトリは既に `mma.sync.aligned.m8n8k4` を直書きしている
（`cuda_dg_kernels_tc.cu:16`）ため、**未取り込みで残っているのは XOR 置換そのもの**。

FP64 では half-warp（16 レーン）ごとに `addr mod 16` が相異なる必要がある。
現行アクセスを解析すると論文 Fig.3 と同型のコンフリクトが出る:

- A オペランド `sD1D[row + (k0+colk)*8]`（`cuda_dg_kernels_tc.cu:101`）
  lane 0–15 で `addr = row + 8·k` → `mod 16` は {row, row+8} の 8 値のみ → **2-way**
- B オペランド `sFluxX[(k0+rowk) + coln*8 + k*64]`（`:102`）→ 同型 → **2-way**
- z 方向 `sFluxZ[ii + jj*8 + (k0+rowk)*64]`（`:129`）
  `64 mod 16 = 0` のため lane 0–15 で `addr mod 16 = coln` の 4 値のみ → **4-way（最悪）**

対処は 2 通りで、併用できる。

1. **A オペランド（`D1D`）はフラグメント順に事前格納する。**
   `sA[k0][row*4 + k]` の並びにすれば lane 0–15 が `0..15` を張り、XOR なしで
   コンフリクトフリー。`D1D` は 64 要素・ロード 1 回なので置換コストは実質ゼロ。

2. **B オペランド（flux 配列）は論文どおり XOR 置換。**
   線形インデックスのみの関数なので、書き込みと読み出しに同じ式を掛ければ整合する。
   机上で導出した候補:

   - `sFluxX` / `sFluxY`: `s = idx ^ (((idx >> 4) & 1) << 2)`
     → coln = 0,1,2,3 が `0-3, 8-11, 4-7, 12-15` に分散
   - `sFluxZ`: `s = idx ^ (((idx >> 6) & 3) << 2)`
     → 実効的に `coln + 4·rowk` となり `0..15` を張る

   論文は「行列ごとに専用パターンの設計が必要」と述べているとおり、単一の XOR で
   3 方向を同時に直すことはできない。本ケースでは **配列が別なので配列ごとに別の
   swizzle を選べる** ことが逃げ道になる。
   エピローグの `sDx[node1]`（`:174`）等の連続アクセスは、XOR が 16 要素内の
   全単射なので壊れない。

   上記の式は机上導出であり未検証。実装前に ncu で
   `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum`
   を採取し、置換前後で比較すること。

### A-2. p=255 手書き TC カーネルの扱い

`tendency_{x,y,z}_p255_tc_kernel`（`cuda_dg_kernels_tc.cu:180` 以降）は
**shared memory を一切使わず**、`D1D` と `q*vel` を毎 mma ごとに global から
直接レジスタへ読んでおり、データ再利用がない。さらに
`<<<8M blocks, 32 threads>>>`（1 ブロック = 1 warp）で、論文の
「1 要素 = 1 スレッドブロック、warp が複数スライス担当」とは逆の構成。
TC 版が tiled FUSED 比 1.13× しか出ない理由はこれで説明できる。

ただし p=255 では既に CUTLASS / cuBLAS 経路が 4.3× 速く
（`overall_summary_report.md` §1）、これは論文の手書きカーネルより成熟した
multistage mainloop である。**手書き TC の p=255 パスに投資する価値は低い。**
廃止候補として明記し、投資は A-1（p=7）と C 系に回すのが妥当。

### A-3. shared memory ルーフラインの導入【解析手法】

論文式 (10)。現行レポートは DRAM ルーフラインのみだが、p=7 FUSED は
DRAM 36% / L1 90% で明らかに shared 律速であり、この指標でないと
「あとどれだけ余地があるか」が判定できない。
GB200 の SM 数・クロックを入れて `overall_summary_report.md` の効率表に一欄追加するのが
最も安価な改善。

### B-1. mma 形状の拡大【論文の先を行く部分】

論文は A100 のため `m8n8k4` しか選択肢がないが、sm_90 以降は FP64 の
`m16n8k4 / m16n8k8 / m16n8k16` が使える。k を深くすれば同じ演算量あたりの
shared 読み出し回数と命令数が減り、A-1 と同じく L1 律速の緩和に効く。
GB200 = sm_100 での利用可否の確認が必要。

**重要な前提差**: 論文の 2.3× のうち 2× 分は「A100 では FP64 Tensor Core peak が
FP64 CUDA-core peak の 2 倍」という事実に由来する。**GB200（Blackwell）では
この 2× アドバンテージが撤廃されており、FP64 Tensor Core peak = FP64 CUDA-core
peak = 40.1 TFLOP/s である**（`overall_summary_report.md` §7 に記録）。したがって本リポジトリで
Tensor Core 化から期待できるのは、演算ピークの向上ではなく
**shared memory / L1 パイプの命令数と wavefront の削減**だけである。
§6 で得た 1.11× はこの範囲の利得であり、論文の 2.3× が再現しないのは想定内。

### B-2. レジスタ圧と shared 使用量のチューニング

p=7 TC は 32 regs / 28.16 KB smem / occ 97% で資源に余裕がある一方、
`sFluxX/Y/Z` と `sDx/Dy/Dz` を別々に確保している
（`cuda_dg_kernels_tc.cu:35-36`）。`sFlux*` を `sD*` に in-place で潰せば
smem は半減し、A-1 と組み合わせて blocks/SM を上げられる。
ビルドに `-Xptxas -v` を足してスピルを常時監視する運用も、そのまま真似できる。

### B-3. 1D 作用素行列の置き場所

論文 Table 2 の結論（shared が最良、constant は ADU 逼迫、texture はスピル）は、
p=7 が既に shared（`sD1D`）である現状維持の裏付けになる。
逆に p=255 は `D1D` を global 直読みしており、A-2 の論点と重なる。

### B-4. 任意サイズへの padding【将来の PolyOrder 拡張時】

現状 N=8 と N=256 はどちらも 8 の倍数なので不要。
p=3, 5 等を TC 化する場合は、論文 §5.3 の
「メモリとスレッドの両方を padding、3D では x0/x1 方向のみ」を採用し、
N=10〜14 の実効効率低下（Fig.10）に注意する。
低次数向けには論文の patch-wise 集約（式 (9), Fig.1）で行列次元を稼ぐ手もあるが、
本リポジトリの要素ごとの契約を変えるため採用は慎重に。

---

## 3. 取り込み不可・要注意

### C-1. FP16 + error correction（論文の 4 倍高速化の本体）

**そのままでは適用できない。** 論文の 4.29× は
「V-cycle を低精度の *前処理* として使い、外側 FGMRES で精度を回復する」構図で、
誤差は反復で吸収される。本リポジトリは陽的 RK による移流の時間積分であり、
誤差を吸収する反復がなく、`AGENTS.md` の
「差は丸め誤差レベル」という検証基準も満たせない
（論文 Fig.8 で FP16 + EC は FP32 相当であって倍精度ではない）。

関連技術として、Algorithm 1 の 2 項分割は cuBLAS の FP64 エミュレーション
（本リポジトリの `CublasEmulation` フラグ）と同じ発想。API は cuBLAS 13.0
update 2 以降で利用できる。旧記述の「現環境では API 不在」はプリプロセッサ
判定の誤りで、詳細は `cublas_emulation_survey.md`。
GB200 の BF16 TC スループットを考えると、**p=255 の巨大 GEMM に対して
多項分割（Ozaki 系）で FP64 相当精度を出す**方向は理屈上は有望。
ただし 2 項では足りず 6〜9 項必要で演算量が一桁増えるため、実装前に
「必要項数 × 実効 BF16 TFLOP/s vs 現行 FP64 GEMM」の見積もりを先に行うこと。

### C-2. WMMA API

論文の結論どおり劣位。本リポジトリは既に MMA 直書きのため、
退行させないこと以外に作業はない。

---

## 4. 推奨する着手順

1. ncu に Memory Workload Analysis セクションを追加し、
   `tendency_fused_p7_tc_kernel` の shared バンクコンフリクトを **まず測る**（現状未計測）。
2. A-1 の swizzle を p=7 TC カーネルに実装。`input_p7_val*.conf` で
   point-varying な `u, v, w, Escale, normal_fn, Fscale` を使い、
   `CUDAFORTRAN_SPLIT` と `dqdt(:,1:Ne)` 全体を突き合わせる。
3. B-2（smem in-place 化）を重ねて再測定。
4. A-3 の shared ルーフラインを解析側に追加し、2〜3 の結果を
   「上限まであとどれだけか」で評価する。
5. その上で B-1（`m16n8k*`）の可否を判断する。

---

## 5. 測定結果: shared memory バンクコンフリクト（§4 手順 1 の実施）

実施日: 2026-08-25 / Slurm job `43554` / スクリプト `job_ncu_bankconflict.sh`
（コミット対象外）。`module load nvhpc-hpcx`、GB200 1 GPU。
条件は `job_all.sh` と同一（p=7, `NeX=NeY=NeZ=32`, `nstep=10`,
`DGOptrKernel_OptType=OPT1`, `dt=1.0D-5`）。
各カーネルの 6 回目の launch を `ncu -s 5 -c 1` で採取。
生成物: `output/FUSED_TC_p7_bank_metrics.csv`,
`output/FUSED_TC_p7_bank_full.ncu-rep{,.txt}`, および `FUSED_p7_*` の同名一式。

### 5.1 測定値

| metric | `tendency_fused_p7_tc_kernel` (TC) | `tendency_fused_p7_kernel` (CUDA core) |
|---|---:|---:|
| duration | 662.2 µs | 549.3 µs |
| shared **load** 命令数 | 7,340,032 | 19,398,656 |
| shared **load** wavefronts | 22,199,947 | 31,553,018 |
| shared **load** バンクコンフリクト | **8,568,459** | 95,738 |
| shared **store** 命令数 | 3,801,088 | 2,228,224 |
| shared **store** wavefronts | 21,094,048 | 4,287,212 |
| shared **store** バンクコンフリクト | **13,885,088** | 223,980 |
| MIO throttle stall (warps/issue) | **19.09** | 2.26 |
| short scoreboard stall | 10.26 | 5.40 |
| long scoreboard stall | 32.88 | 21.06 |
| SM throughput | 22.62% | 38.65% |
| L1/TEX throughput | 80.78% | 88.86% |

ncu 自身の診断（`FUSED_TC_p7_bank_full.ncu-rep.txt`）:

- shared load: 平均 **3.0-way** コンフリクト、全 wavefront の 38.59% がコンフリクト由来。
  `OPT Est. Speedup: 31.47%`
- shared store: 平均 **5.8-way** コンフリクト、全 wavefront の 65.78% がコンフリクト由来。
  `OPT Est. Speedup: 53.64%`
- 合計で 17,825,792 wavefront（全 38,666,240 の **46%**）が余剰。
- Scheduler: `No Eligible` 80.48%、issue は 5.1 サイクルに 1 回。

### 5.2 §2 A-1 の予測との突き合わせ

- **論文の "MMA basic" 状態にあるという診断は裏づけられた。**
  MIO throttle stall が 19.09 対 2.26（CUDA core 版）で、論文が MMA basic → MMA CF で
  11.58 → 2.11 に落としたという記述と同じ構図。
- **load 側の 2-way 予測は保守的すぎた。実測は 3.0-way。**
  §2 A-1 で 2-way と見積もった A/B オペランド読み出しに加え、
  z 方向 `sFluxZ`（`cuda_dg_kernels_tc.cu:129`、机上では 4-way）が効いており、
  全体平均が 3.0-way に押し上げられている。
- **最大の問題は store 側（5.8-way）で、これは §2 A-1 で見落としていた。**
  mma アキュムレータの書き戻し

  ```
  sDx[i_c + j0_c * 8 + k * 64] = c0;      // cuda_dg_kernels_tc.cu:106-107
  sDx[i_c + (j0_c + 1) * 8 + k * 64] = c1;
  ```

  は `i_c = lane>>2`, `j0_c = (lane&3)*2` なので `addr = row + 16*colk`。
  FP64 では half-warp 内で `addr mod 16` が相異なる必要があるが、
  `16*colk mod 16 = 0` のため lane 0–15 の `addr mod 16` は `row`（0..3）の
  **4 値しか取らない → 4-way**。`c1` 側も `row+8` で同じく 4-way。
  `sDz` の書き戻し（`:133-138`）はさらに散っており、平均 5.8-way はこれで説明できる。
  m8n8k4 の C フラグメントは lane あたり 2 要素が列方向に隣接するため、
  **本来は `double2` の 16 B ストア 1 回にまとめられる**（現在は 8 B ストア 2 回）。

- **TC 化による shared 命令削減は、コンフリクトで完全に食い潰されている。**
  shared load 命令数は TC 版が CUDA core 版の 0.38 倍（7.34M 対 19.40M）と
  期待どおり減っているのに、wavefront 合計は 43.3M 対 35.8M で **TC 版のほうが多い**。
  TC 版が 1.21× 遅い直接の原因はこれ。

- `sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active` は
  sm_100 で `n/a`。Blackwell では別名の metric を探す必要がある（B-1 の判断材料）。

### 5.3 これを受けた A-1 の作業内容の更新

優先順位を **store 側を先** に入れ替える。

1. **アキュムレータ書き戻しの `double2` 化 + 出力レイアウトの swizzle**（推定効果最大、
   ncu 見積り 53.6%）。`c0`/`c1` が隣接するよう `sDx/sDy/sDz` の格納順を組み替え、
   `*((double2*)(out + idx)) = c` の形にする。論文 Appendix B の Listing 4 が
   まさにこの形（`*((double2*)(out + c_idx)) = c[z]` と XOR 置換の併用）。
   → **この予測は外れた。§6.3 を参照**。出力レイアウトの組み替えは効いたが、
   `double2` 化そのものは GB200 では逆に遅い。
2. **z 方向 `sFluxZ` の読み出し swizzle**（`s = idx ^ (((idx>>6)&3)<<2)`、4-way → 1-way）。
3. **A/B オペランド読み出しの整理**（`D1D` のフラグメント順事前格納、
   `sFluxX/sFluxY` の `s = idx ^ (((idx>>4)&1)<<2)`）。

ncu の見積りを単純合成すると 1.3〜1.5× 程度で、
現状 1.21× 負けている CUDA core 版を上回る余地がある。
実装後は `input_p7_val*.conf` で point-varying な
`u, v, w, Escale, normal_fn, Fscale` を与え、`CUDAFORTRAN_SPLIT` と
`dqdt(:,1:Ne)` 全体を突き合わせて検証すること。

---

## 6. 実装結果: p=7 Tensor Core カーネルの shared レイアウト刷新

実施日: 2026-08-25。対象 `cuda_dg_kernels_tc.cu` の `tendency_fused_p7_tc_kernel`
（未コミット）。測定は Slurm job `43554`（実装前）/ `43573` / `43601` / `43612` /
`43616` / `43617` / `43618`（実装後）。条件は §5 と同一。

### 6.1 採用した変更

1. **1D 微分行列を m8n8k4 フラグメント順で保持**（`sD1D` → `sDfrag`）。
   `sDfrag[b*32 + r*4 + c] = D1D[r + (b*4+c)*8]` とすることで、
   lane L の読み出しが 16 連続 double を張り、XOR なしでコンフリクトフリー。
   x/y/z の 3 方向すべてがこの 1 本の配列を共有する。
2. **`sFluxX` / `sFluxY` に XOR 置換** `sw_xy(idx) = idx ^ (((idx>>4)&1)<<2)`。
   インデックスの bit4 を、常に空いている bit2 に畳み込む。
3. **`sFluxZ` に XOR 置換** `sw_z(idx) = idx ^ (((idx>>6)&3)<<2)`。
   z 方向の縮約は 64 double ストライドなので bit6-7 を bit2-3 に畳み込み、
   4-way を解消する。
4. **`sDx` / `sDy` を面内転置 + XOR** `idx_dxy(i,j,k) = 64k + 8i + (j ^ i)`。
   m8n8k4 のアキュムレータは `C[i][2c]`, `C[i][2c+1]` を保持し `i = lane>>2`
   なので、自然順 `i + 8j` ではストアが 1 フェーズあたり 4 バンクしか叩かない。
5. **`sDz` に XOR 置換** `sw_dz(idx) = idx ^ (((idx>>6)&1)<<3)`。

### 6.2 測定結果

| | 実装前 | 実装後 | CUDA core 版 |
|---|---:|---:|---:|
| kernel duration (ncu) | 662,240 ns | **497,248 ns** | 549,760 ns |
| shared load バンクコンフリクト | 8,568,459 | **194,876** | 93,734 |
| shared load wavefronts | 22,199,947 | 13,826,364 | 31,551,014 |
| shared store バンクコンフリクト | 13,885,088 | 7,817,092 | 226,369 |
| shared store wavefronts | 21,094,048 | 15,026,052 | 4,289,601 |
| MIO throttle stall | 19.09 | **1.77** | 2.26 |
| long scoreboard stall | 32.88 | 23.59 | 21.08 |
| SM throughput | 22.62% | 30.06% | 38.71% |

`nstep=1000`, `Ne=32^3` の end-to-end（job 43618）:

| `DqdtKernel_Type` | Main | Cal_tend | CUDA device fused |
|---|---:|---:|---:|
| `CUDAFORTRAN_FUSED_TC` | **1.614** | **1.101** | **1.068** |
| `CUDAFORTRAN_FUSED` | 1.692 | 1.182 | 1.150 |

- カーネル単体で **1.33×** 高速化。
- **p=7 で Tensor Core 版が初めて CUDA core 版を上回った**（device 時間で 1.08×）。
  [`overall_summary_report.md`](overall_summary_report.md) §4 の「p=7 では FUSED が最速」
  「FUSED_TC は 1.28× 遅い」という記載は、この変更で覆った。同節の表は commit
  `299a868` 時点の測定なので未更新。
- 数値検証: `SCALE_DG_VARYING_COEFF=1` で `u,v,w,Escale,normal_fn,Fscale` を
  point-varying にし、`CUDAFORTRAN_SPLIT` と `dqdt(:,1:Ne)` 全 4096 点を比較。
  max abs diff `1.776e-15`（`max|dqdt| = 8.18` に対し約 1 ulp）、相対 L2 `8.9e-17`。
- レジスタ 32 本 / スピル 0 / smem 28160 B は変更前と同じ。

### 6.3 採用しなかった案とその実測

論文 Appendix B の Listing 4 は、アキュムレータ対を
`*((double2*)(out + c_idx)) = c[z]` と **16 B ストア 1 回**で書き戻している。
本カーネルでも同じ形にでき、しかもストアのバンクコンフリクトは完全に消えるが、
**実測では遅い**:

| 版 | duration | store wavefronts | store コンフリクト | MIO throttle |
|---|---:|---:|---:|---:|
| 採用版（8 B ストア 2 回、store は 2-way 残り） | **497,248 ns** | 15,026,052 | 7,817,092 | **1.77** |
| `double2` 16 B ストア 1 回（store コンフリクト 0） | 596,544 ns | 11,378,124 | 4,169,164 | 11.16 |
| 8 B ストア 2 回 + store もコンフリクトフリーにする置換 | 596,192 ns | 11,593,338 | 4,384,378 | 11.16 |

- 3 番目の版は `perm_row(i) = (i & ~1) | ((i>>1)&1)` を使い、ストアとエピローグ
  読み出しの両方を同時にコンフリクトフリーにしたもの。この置換では c0/c1 の
  アドレスが隣接するため ptxas が STS.128 に再マージしてしまい、
  inline PTX `st.shared.f64` でマージを抑止した版（`asm volatile` +
  `"memory"` あり / なしの両方）も測ったが、いずれも 596〜599 ns 台で改善しなかった。
- **wavefront が減っても速くならない**という、論文の想定とは逆の挙動である。
  MIO throttle が 1.77 → 11.16 に跳ね上がることから、GB200 (sm_100) では
  128 bit shared ストア、あるいは 1 フェーズ内でレーン順が単調でない
  8 B ストアの発行がボトルネックになっている可能性がある。**未解明**。
  論文は A100 での結果なので、この差は世代依存かもしれない。
- 世代差はピーク性能にも効いている。GB200 では
  **FP64 Tensor Core peak = FP64 CUDA-core peak = 40.1 TFLOP/s** であり
  （`overall_summary_report.md` §7）、A100 にあった 2× のアドバンテージがない。
  論文の 2.3× のうち 2× 分はそのアドバンテージ由来なので、本リポジトリで
  期待できるのは shared / L1 パイプの削減分だけである。

### 6.4 残っている作業

- §2 B-2（`sFlux*` を `sD*` に in-place 化して smem 28 KB → 半減）。
- §2 A-3（shared memory ルーフラインの導入）。§6.2 で L1 throughput は
  90.99% に達しており、次の律速がどこかを判定する材料が要る。
- §2 B-1（`m16n8k*` FP64 mma）。
- 6.3 の異常（wavefront 減 = 遅くなる）の原因特定。マイクロベンチで
  8 B / 16 B shared ストアの発行コストを単独測定するのが早い。
- グローバルロードが uncoalesced（1 セクタ 32 B のうち平均 25.4 B しか使用）で
  `long scoreboard` stall が 23.59 と最大の stall 要因になっている。
  次の一手はここ。

---

## 7. 実装結果: occupancy を 100% にする shared 削減と、分離可能 lift

実施日: 2026-08-25。§6 の続き。対象は同じ `tendency_fused_p7_tc_kernel`。
測定は Slurm job `43734`（§6 実装後 = commit `e971ba5` の状態）/ `43954` / `43959`、
および login node 上の `nstep=1000` 実測。条件は §5 と同一
（p=7, `NeX=NeY=NeZ=32`, `dt=1.0D-5`, `OPT1`）。
プロファイル用に `cuda_dg_kernels_tc.cu` のみ `-lineinfo` を足してビルドした
（生成コードは変わらない）。生成物は `output/*_glob*_{metrics.csv,full.ncu-rep}`。

### 7.1 §6.4 の「次の一手」を測った結果

§6.4 は global load が uncoalesced（1 セクタ 25.4 B）で `long scoreboard` が
最大の stall だから次はそこだ、と書いていた。job `43734` で内訳を取ると、
**その診断は的を外していた**。

| | `e971ba5` の TC カーネル |
|---|---:|
| global load 命令数 | 12,582,912 |
| global load requests | 12,386,304 |
| global load sectors | 120,717,312 |
| sectors / request | **9.75**（完全 coalesce の FP64 warp ロードは 8.0）|
| bytes per sector | 79.48% |
| DRAM read | 1.461 GB = **44.6 KB / element** |
| L1/TEX throughput | 90.92% |
| **Theoretical occupancy** | **75%**（Block Limit Registers 6 / Block Limit Shared Mem 6）|
| Achieved occupancy | 72.11% |

- 1 要素あたりの DRAM 読み出し 44.6 KB は、契約上必ず読む量
  （`q,u,v,w` 16 KB + `Escale` 12 KB + `normal_fn` 9 KB + `Fscale` 3 KB +
  `VMapM/P` 3 KB ≒ 43 KB）とほぼ一致する。**データ量はすでに下限**で、
  減らす余地は無い。
- sectors/request は 9.75 で、理想の 8.0 に対し 1.22× でしかない。
  gather（`VMapM/VMapP` 経由の 96 命令）は 1 命令あたり約 16 セクタだが、
  全体に占める超過は 772/3684 セクタ ≒ 21% にとどまる。
  §6.4 が言うほど支配的ではない。
- **見落としていた本命は occupancy だった。** §6.2 は「レジスタ 32 本 /
  smem 28160 B は変更前と同じ」と書いたが、これは §5 の**変更前**カーネルの
  値である。`e22dda1` 後の実機値は **40 レジスタ**で、しかも
  shared memory carveout が 200.70 KB なので
  smem 28.16 + driver 1.02 = 29.18 KB → 6 ブロック、
  レジスタも 65536/(40×256) → 6 ブロック。
  両方が同時に 6 で頭打ちし、theoretical occupancy は 75% しかなかった。
  ncu 自身も `Est. Speedup: 9.245%` を出していた。

### 7.2 採用した変更

1. **`sDx/sDy/sDz` を `sFluxX/sFluxY/sFluxZ` に in-place 化**（§2 B-2）。
   warp `k` は `sFluxX`/`sFluxY` の平面 `k` しか読まず、`sw_xy()` も
   `idx_dxy()` も平面をまたがないので、mma ループとストアの間の
   `__syncwarp()` で足りる。z だけは `sw_z()` と `sw_dz()` が warp の担当列を
   またぐので `__syncthreads()` が要る。
   smem 28160 B → **15872 B**。
2. **`__launch_bounds__(256, 8)`**。レジスタが 32 本に収まり
   （spill 0）、1 と合わせて **8 ブロック / SM = theoretical occupancy 100%** になる。
3. **`Lift_mat`(512×6) を `Lift1D`(8×6) に置き換え、shared に 48 個だけ置く**。
   `Lift_mat(i,j,k,f)` は 1 つの体積添字だけに依存する（face 1,3 は j、
   face 2,4 は i、face 5,6 は k）。`mod_mesh.f90` は既にこの形の `Lift1D` を
   p=255 と GEMM 経路向けに作っている。global load 命令が 12.58 M → 9.31 M、
   sectors が 120.7 M → 95.9 M（**-20.5%**）になる。

### 7.3 測定結果

ncu 単発カーネル（`-s 5 -c 1`、`--set full`）:

| | `e971ba5` | +in-place +launch_bounds | +`Lift1D` | CUDA core 版 |
|---|---:|---:|---:|---:|
| duration | 501.1 µs | 441.8 µs | **433.1 µs** | 546.9 µs |
| registers / thread | 40 | 32 | 32 | — |
| static smem / block | 28.16 KB | 15.87 KB | 16.26 KB | — |
| Block Limit Registers | 6 | 8 | 8 | — |
| Block Limit Shared Mem | 6 | 8 | 9 | 8 |
| Theoretical occupancy | 75% | **100%** | **100%** | — |
| Achieved occupancy | 72.11% | 96.47% | **96.85%** | 59.98% |
| global load sectors | 120,717,312 | 120,717,312 | **95,944,704** | 120,717,312 |
| L2 read sectors | 137,480,874 | 128,210,645 | 122,654,563 | 123,637,124 |
| L1/TEX throughput | 90.92% | 96.29% | 95.85% | 88.59% |
| DRAM read | 1.461 GB | 1.464 GB | 1.465 GB | 1.462 GB |
| long scoreboard stall | 23.65 | 27.00 | 25.39 | 21.14 |

`nstep=1000`, `Ne=32^3` の end-to-end（login node、各 3 回、ばらつきは
最終桁で ±0.3%）:

| `DqdtKernel_Type` | Main | Cal_tend | CUDA device fused |
|---|---:|---:|---:|
| `CUDAFORTRAN_FUSED_TC`（本節の変更後） | **1.415** | **0.890** | **0.851** |
| `CUDAFORTRAN_FUSED_TC`（`e971ba5`） | 1.646 | 1.128 | 1.076 |
| `CUDAFORTRAN_FUSED`（対照） | 1.714 | 1.190 | 1.153 |

- device 時間で `e971ba5` 比 **1.26×**、CUDA core 版比 **1.35×**。
- ncu 単発時間（501 → 433 µs、1.16×）より end-to-end の改善が大きい。
  ncu はリプレイのたびに L2 を流すので、RK ステージ間で `q,u,v,w` が L2 に
  残る実運用より DRAM 側を重く見積もる。実測の 1 launch = 359 µs（変更前）
  → 284 µs（変更後）に対し、ncu は 501 → 433 µs である。
  **この 2 つは別物として扱うこと。**
- 数値検証: `SCALE_DG_VARYING_COEFF=1` で `u,v,w,Escale,normal_fn,Fscale` を
  point-varying にし、`CUDAFORTRAN_SPLIT` と `dqdt(:,1:Ne)` 全点を比較。
  `Ne=8^3, nstep=3` で max abs diff `5.618e-14`（`max|dqdt|=8.52`）、
  相対 L2 `4.70e-16`。`Ne=16^3, nstep=5` で max abs diff `1.119e-13`、
  相対 L2 `1.03e-15`。いずれも変更前の TC カーネルと同じ差であり、
  in-place 化と `Lift1D` 化は当該入力で**変更前とビット一致**した。
- 非 CUDA ビルド（`make clean && make`）も通ることを確認した。

### 7.4 効かなかった / 逆効果だった案（実測値）

いずれも `nstep=1000`, `Ne=32^3` の `CUDA device fused`（秒）。

| 版 | device 時間 |
|---|---:|
| `e971ba5`（基準） | 1.076 |
| `Lift1D` を shared に置く**だけ**（occupancy 6 ブロックのまま） | 1.090（**1.3% 遅い**）|
| `Lift1D` を global から直接読む（`Lift1D[j]` など） | 1.318 |
| `Lift_mat` のまま、face 1-4 の係数を node1 のものと共用（12→8 ロード） | 1.369 |
| in-place + `__launch_bounds__` | 0.870 |
| in-place + `__launch_bounds__` + `Lift1D` | **0.851** |

読み取れること:

- **global セクタを 20% 減らしても、6 ブロックのままでは遅くなる。**
  epilogue は L1/TEX で律速しており、L1/TEX は shared ロードと global ロードを
  同じパイプで捌く。global を shared に移し替えても総量は減らない。
  同じ置き換えが 8 ブロックでは 2.2% の利得になるので、
  **この最適化は occupancy を上げた後でなければ評価できない**。
- `Lift1D` を global から直読みすると warp 内で添字 `j` が 8 通りに散り、
  coalesce していた 12 本のロードが gather 12 本に化ける。
  `Lift_mat` 側は `node1 = tid` で完全 coalesce だったので、
  「行列を小さくする」こと自体には価値がない。
- 12→8 ロードに減らすだけの版が 27% 遅くなった理由は未解明。
  ロード本数ではなくスケジューリングが効いている。

### 7.5 残っている作業

- 現在の律速は L1/TEX 95.9%、`long scoreboard` 25.39、
  DRAM は 46.5%（3.69 TB/s）。実運用では 1 launch 284 µs で
  必須トラフィック約 1.54 GB を動かしており、実効 5.4 TB/s。
  GB200 の HBM ピーク（約 8 TB/s）に対し 68% で、ここから先は
  帯域効率の勝負になる。
- M 側の gather（`q,u,v,w[iM]`、96 gather 命令のうち 48 本）は、
  `iM` が自要素内のノードなので `q,u,v,w` を shared に置けば消える。
  ただし +16 KB で 8 ブロックを維持できない（16.26 → 32 KB 超）。
  §7.4 の教訓どおり、**occupancy を落とす形での L1 削減は評価に値しない**。
  やるなら `sflux_bnd` の圧縮など、他で smem を取り戻してから。
  → **§9 で実測、不採用**（2026-08-25）。掛け算の順序を入れ替えれば +4 KB で
  occupancy を落とさずに実装できたが、global sectors −26% と引き換えに
  face gather の shared バンクコンフリクトが 20 倍になり、L1/TEX は
  95.4% のまま動かず 3.7% 遅くなった。
- §2 B-1（`m16n8k*` FP64 mma）は未着手。ただし `Compute (SM)` は 32% で、
  演算パイプは余っている。優先度は低い。
- §6.3 の異常（wavefront 減 = 遅くなる）と §7.4 最終行の異常は、
  どちらも「命令数・wavefront 数を減らしたのに遅い」形をしている。
  マイクロベンチで shared ストアと発行スケジューリングを単独測定するのが早い。

---

## 8. §7 の知見を CUDA core 版に適用する

実施日: 2026-08-25。§7 の 3 つの変更のうち 2 つは Tensor Core と無関係で、
`tendency_fused_p7_kernel`（CUDA core 版、`mod_cuda_dg_kernels.cuf`）にも
そのまま効く。測定は Slurm job `44039` と login node の `nstep=1000` 実測、
条件は §5 と同一。

### 8.1 適用可否

| §7 の変更 | CUDA core 版への適用 |
|---|---|
| `sD*` の `sFlux*` への in-place 化 | **不要**。CUDA core 版は微分和をレジスタ（`sum_x1` 等）に貯めるので `sDx/sDy/sDz` を持たず、既に smem 15.87 KB で `Block Limit Shared Mem = 8` に達していた。 |
| occupancy 100% 化 | **適用可**。ただし律速していたのは smem ではなく**レジスタ 42 本**で、`Block Limit Registers = 5`、theoretical occupancy 62.5% だった。 |
| `Lift_mat` → `Lift1D` | **適用可**。epilogue の形は TC 版と同一。 |

### 8.2 CUDA Fortran での launch bounds

`attributes(global,launch_bounds(256,8))` は nvfortran 26.3 で構文エラーになる。
正しい書き方は**属性リストの外**に置く形:

```fortran
attributes(global) launch_bounds(256,8) subroutine tendency_fused_p7_kernel( &
```

`-gpu=maxregcount:32` でも同じ 32 レジスタになるが、こちらは**コンパイル単位
全体**に効く。実測すると p=255 の `CUDAFORTRAN_GEMM` が 0.403 → 0.425 秒
（+5.5%）と悪化したので採用しない。per-kernel の `launch_bounds` では
p=255 の 4 経路すべてが測定誤差内で不変であることを確認した。

### 8.3 測定結果

`nstep=1000`, `Ne=32^3` の `CUDA device fused`（秒、各 3 回）:

| 版 | device 時間 |
|---|---:|
| `e971ba5`（基準） | 1.153 |
| `-gpu=maxregcount:32` のみ | 1.024 |
| `Lift1D` のみ（レジスタ 42 → 40、5 → 6 ブロック） | 1.010 |
| **`Lift1D` + `launch_bounds(256,8)`** | **0.986** |

ncu 単発カーネル（job `44039`）:

| | `e971ba5` | +`Lift1D`+`launch_bounds` |
|---|---:|---:|
| duration | 550.1 µs | 509.5 µs |
| registers / thread | 42 | 32（spill 0）|
| static smem / block | 15.87 KB | 16.26 KB |
| Block Limit Registers | 5 | 8 |
| Theoretical occupancy | 62.5% | **100%** |
| Achieved occupancy | 60.04% | **96.58%** |
| global load sectors | 120,717,312 | 95,944,704 |
| L1/TEX throughput | 88.54% | 95.49% |
| Compute (SM) throughput | 38.52% | 41.42% |

- device 時間で **1.17×**。
- 数値検証: `SCALE_DG_VARYING_COEFF=1` で `Ne=8^3, nstep=3` と
  `Ne=16^3, nstep=5` の両方について `CUDAFORTRAN_SPLIT` と
  `dqdt(:,1:Ne)` 全点が**ビット一致**した。変更前の `CUDAFORTRAN_FUSED` とも一致。
- 非 CUDA ビルドも通ることを確認した。

### 8.4 p=7 の順位

| `DqdtKernel_Type` | Main | Cal_tend | CUDA device |
|---|---:|---:|---:|
| `CUDAFORTRAN_FUSED_TC` | **1.415** | **0.890** | **0.852** |
| `CUDAFORTRAN_FUSED` | 1.549 | 1.024 | 0.986 |

TC 版が最速という §7 の結論は変わらないが、差は 1.35× から **1.16×** に縮んだ。
§7 で「TC 版が CUDA core 版より 1.35× 速い」と書いた部分は、
CUDA core 版が occupancy 60% の状態と比べたものである。
**両者を 100% occupancy で揃えた比較が上表であり、以後はこちらを使うこと。**

### 8.5 一般化できる教訓

- **occupancy を先に確認する。** 本リポジトリの p=7 カーネルは 2 つとも
  「shared / L1 が 90% 前後だから L1 律速だ」と読める状態でありながら、
  実際にはどちらも occupancy（TC 版は smem とレジスタ、CUDA core 版は
  レジスタ単独）で 60〜75% に抑えられていた。
  ncu の `Block Limit *` 行を読まずに Memory Workload Analysis だけを見ると
  この診断を落とす。§6.4 で「次は global load」と書いたのがその例。
- **L1 削減の評価は occupancy を上げた後で行う。** `Lift1D` 化は
  6 ブロックの TC 版では 1.3% 遅く、8 ブロックでは 2.2% 速い（§7.4）。

---

## 9. M 側 gather を shared に移す案（不採用）

実施日: 2026-08-25。§7.5 が「次の一手」として挙げていた案の実測である。
対象は `tendency_fused_p7_tc_kernel`、条件は §5 と同一
（p=7, `NeX=NeY=NeZ=32`, `dt=1.0D-5`, `OPT1`, `nstep=1000`, login node）。
ncu は Slurm job `44568`、ベースは commit `e910708`（device 0.852 秒）。

### 9.1 shared 予算の問題は解ける

§7.5 は「`q,u,v,w` を shared に置けば M 側 gather 48 本が消えるが +16 KB で
8 ブロックを維持できない」と書いていた。これは順序を変えれば回避できる。

数値流束の M 側は `qM` と `VelM = uM*nx + vM*ny + wM*nz` を別々に必要とするが、
必要な値は**体積積の計算ですでに coalesce してロードしている** `q,u,v,w`
そのものである。そこで prologue で `q*u` を作らず、

1. `sFluxX/Y/Z` に `u,v,w` を**そのまま**置き、`q` は別配列 `sQvol[512]` に置く
   （+4 KB のみ）。
2. face 点スレッドは M 側を shared から読む（`iM` は `VMapM` の定義上つねに
   自要素内なので、局所添字は `VMapM[fidx]-1-elem_offset`）。
3. その後で `sFlux*` を `q` 倍して体積フラックスにする。

とすれば、追加は `sQvol` の 4 KB だけで済む。実際 smem 16.26 → 20.35 KB、
レジスタ 32 本（spill 0）、8 ブロック / SM、achieved occupancy 96.7% が
維持される。**occupancy を落とさずに M 側 gather を消せる**ので、
§7.4 の交絡（occupancy が下がったせいで遅い）は今回は無い。

3 の掛け算には 2 通り試した。

- **版 A**: 独立パスで `sFluxX[p] *= sQvol[n]` を行う。
- **版 B**: x/y は mma のオペランドロード時に掛ける。x/y の各フラックス要素は
  contraction 中ちょうど 1 回しか読まれないので命令は増えず、`sQvol` を
  `sw_xy()` 順に置けばフラックスと同じアドレスになりコンフリクトも増えない。
  z だけは warp をまたぐ別の置換なので独立パスで掛ける。

### 9.2 測定結果

`nstep=1000`, `Ne=32^3`（各 3 回）:

| 版 | CUDA device fused | Main |
|---|---:|---:|
| ベース（`e910708`） | **0.852** | **1.211** |
| 版 A（独立の掛け算パス） | 0.866（+1.7%） | 1.226 |
| 版 B（mma オペランドに畳み込み） | 0.858（+0.7%） | 1.217 |

ncu（job `44568`、`-s 5 -c 1`、ベース対 版 B、1 launch）:

| | ベース | 版 B | 比 |
|---|---:|---:|---:|
| duration | 432.99 µs | 448.90 µs | 1.037 |
| global load 命令 | 9,306,112 | 7,733,248 | **0.831** |
| global load sectors | 95,944,704 | 70,778,880 | **0.738** |
| L2 read sectors | 122,468,031 | 102,789,847 | 0.839 |
| DRAM read | 1.465 GB | 1.474 GB | 1.006 |
| shared load 命令 | 9,437,184 | 13,107,200 | 1.389 |
| shared load wavefronts | 16,479,208 | 28,211,683 | **1.712** |
| shared load bank conflicts | 226,280 | 4,618,723 | **20.41** |
| shared store wavefronts | 13,665,657 | 16,390,655 | 1.199 |
| L1/TEX throughput | 95.69% | 95.43% | 0.997 |
| SM throughput | 32.38% | 35.51% | 1.097 |
| long scoreboard stall | 25.48 | 17.34 | 0.681 |
| mio throttle stall | 5.26 | 7.97 | 1.515 |
| barrier stall | 7.29 | 6.30 | 0.864 |
| achieved occupancy | 96.70% | 96.68% | 1.000 |

### 9.3 結論と、それでも残る含み

**global 側の狙いは当たっている**。gather 命令 −17%、sectors −26%、
L2 read −16%、`long scoreboard` 25.48 → 17.34 と、§7.5 が期待した通りに
global のレイテンシ圧力は下がった。それでも遅いのは、**L1/TEX が
95.4% で張り付いたまま動かない**からである。減った global sectors の代わりに
shared load wavefronts が +1.17 千万発生し、L1/TEX の総量は変わらなかった。

wavefront 増の内訳は命令数ではなくバンクコンフリクトである。shared load
命令は +389 万（1 ブロックあたり約 112 本）しか増えていないのに、
コンフリクトは 22.6 万 → 462 万と **20 倍**になった。原因は face 点の
gather パターンである。face 2（`i=7` 面）の 16 レーンが読むノードは
`7 + 8j + 64k` で、**アドレス差がすべて 8 の倍数**になるため、
`sw_xy()` を通しても 16 レーンが 4 バンクにしか散らない。
face 1（`j=0` 面、`i + 64k`）も 2-way になる。
一方 `sw_xy()` / `sw_z()` は contraction のアクセスパターンに合わせて
設計されたものなので、face gather を同時に平坦化はしない。

したがって **この案は「L1/TEX 律速を global→shared の付け替えで解消できる」
という前提そのものが成り立たない**（§7.4 の教訓の再確認）。ただし今回は
occupancy が落ちていないので、§7.4 とは別の理由で負けている点に注意。

含みとして残るのは次の 1 点である。**contraction のアクセスと face gather の
両方を同時にコンフリクトフリーにする shared レイアウトが見つかれば、
この案は勝ちうる**。global を 26% 減らして 3.7% しか負けていないので、
462 万のコンフリクトを 22.6 万の水準に戻せれば符号は反転する見込みが高い。
ただしそれは「1 つの置換で 3 種類の contraction パターンと 6 面の gather
パターンを同時に満たす」探索問題であり、面ごとに置換を変えられない以上、
自明な XOR swizzle の範囲では解が無い可能性が高い。着手するなら
レイアウト探索を単独の小問題として（マイクロベンチで）先に解くこと。

→ **§10 で解いた。この段落の予想は誤りである**（2026-08-25）。解は 3072 個
存在し、実機でもコンフリクトを 89% 消して L1/TEX を 95% の壁から外す。
それでもなお遅く、理由は L1/TEX ではなく整数・アドレス演算だった。

コードはベース（`e910708`）に戻してある。

---

## 10. shared レイアウトのマイクロベンチ: 律速は L1/TEX だけではなかった

実施日: 2026-08-25。§9 の宿題（「contraction と face gather を同時に
コンフリクトフリーにする置換が存在するか」）と、§6.3 / §7.4 の未解明な異常
（「命令数・wavefront を減らしたのに遅い」）を、同じ道具で片付けた。
ncu は Slurm job `44592`、end-to-end は login node の `nstep=1000`、
条件は §5 と同一。ベースは commit `e910708`。

### 10.1 アクセスパターンの厳密モデル

まずカーネルの shared 命令を 1 本ずつ（warp × lane の関数として）列挙した
モデルを作り、ncu と突き合わせた。**命令数は完全一致**する。

| | モデル | ncu（1 ブロックあたり） |
|---|---:|---:|
| shared load 命令 | 288 | 9,437,184 / 32768 = 288 |
| shared store 命令 | 112 | 3,670,016 / 32768 = 112 |

コンフリクトも一致する。ただし**同一アドレスはブロードキャストで無償**
（バンクコンフリクトは「同じバンクの*異なる*アドレス」）である点を入れないと
`sLift[j]` のような 8 レーン同アドレスの読み出しを誤って数える。
これを入れると、コミット版の load コンフリクトはモデル 0 に対し実測
22.5 万（6.9 / ブロック）、§9 の版はモデル 128 / ブロックに対し実測
141 / ブロックで、いずれも一致する。

### 10.2 置換は存在する（§9 の予想は誤り）

決め手は、**この カーネルの全アクセスが、16 レーンの 1 フェーズ内では
ノード添字 `n = i + 8j + 64k` の座標部分空間のコセットになっている**ことである
（`i` は bit 0-2、`j` は bit 3-5、`k` は bit 6-8）。

| アクセス | フェーズ内で動くビット |
|---|---|
| x / y contraction | `i0,i1,j0,j1` |
| z contraction | `i0,i1,k0,k1` |
| face 1,3 の M gather | `i0,i1,i2,k0` |
| face 2,4 の M gather | `j0,j1,j2,k0` |
| face 5,6 の M gather / 線形ストア | `i0,i1,i2,j0` |

アドレス下位 4 ビットを GF(2) 線形写像とみなし、ビット `b` の像を `c_b` と書くと、
あるパターンがコンフリクトフリーである条件は
**そのパターンの 4 本の `c_b` が GF(2) 上で一次独立**、それだけになる。
XOR swizzle（下位ニブルは恒等、上位ビットの関数を XOR）に限って全数探索すると、
**6 パターンすべてを満たす swizzle が 65536 通り中 3072 通り存在する**。
採用したのは

```
addr = n ^ (((b4 ^ b7) << 2) | (b5 << 1) | (b6 * 0b1001))
```

で、フル命令モデルに掛けると M 側 staging 版の load コンフリクトは
**128 → 0 / ブロック**になる。コミット版の `sw_xy` は z contraction と
face 1-4 gather で、`sw_z` は x/y contraction と face 1-4 gather で
それぞれ条件を満たしていない。

### 10.3 ところが実機は遅くなる

| 版 | swizzle | M 側 | CUDA device (`nstep=1000`) |
|---|---|---|---:|
| `e910708`（ベース） | `sw_xy`/`sw_z` | global gather | **0.852** |
| §9 の版 B | `sw_xy`/`sw_z` | shared | 0.858 |
| 新 swizzle のみ | 新 `sw` | global gather | 0.866 |
| 新 swizzle + M 側 staging | 新 `sw` | shared | 0.899 |

ncu（job `44592`、`-s 5 -c 1`、1 launch）:

| | ベース | 版 B | swizzle のみ | swizzle + staging |
|---|---:|---:|---:|---:|
| duration | 433.8 µs | 450.3 µs | 434.0 µs | 445.7 µs |
| shared ld bank conflicts | 224,500 | 4,617,000 | **224,000** | **505,500** |
| shared ld wavefronts | 1.648e7 | 2.821e7 | 1.648e7 | 2.410e7 |
| L1/TEX throughput | 95.58% | 95.40% | 95.84% | **92.53%** |
| global load sectors | 9.594e7 | 7.078e7 | 9.594e7 | 7.078e7 |
| long scoreboard stall | 25.48 | 17.41 | 22.82 | 17.86 |
| mio throttle stall | 5.29 | 8.00 | 4.37 | 5.04 |
| **ALU pipe 命令** | **2.464e7** | 3.277e7 | 3.041e7 | **3.749e7** |
| 整数命令（thread） | 1.227e9 | 1.445e9 | 1.479e9 | 1.706e9 |
| 全命令（warp） | 8.857e7 | 1.006e8 | 9.670e7 | 1.105e8 |
| SM throughput | 32.34% | 35.48% | 33.78% | 37.94% |
| achieved occupancy | 96.71% | 96.63% | 96.67% | 96.53% |

**モデルの予測はハードウェア上で当たっている**。新 swizzle 単体はコミット版と
コンフリクト数・wavefront 数が同一（22.4 万 / 1.648e7）で、モデルが
「コンフリクト的に等価」と言った通り。staging と組み合わせると
コンフリクトは 462 万 → 50.6 万（**−89%**）、wavefront は 2.82e7 → 2.41e7、
そして **L1/TEX がついに 95% の壁から外れて 92.53%** になる。
§9 が「符号が反転する見込みが高い」と書いた条件は満たされた。

**それでも遅い。** 4 版を通して唯一きれいに効いている説明変数は
**整数・アドレス演算の量**である。ALU pipe 命令はベース 2.46e7 に対し
staging 版で 3.75e7（**+52%**）、swizzle 単体でも +23%。
ベース時点ですでに全 thread 命令の 43% が整数演算で、8 ブロック / SM・
occupancy 96.7% で回っているこのカーネルでは、発行スロットが
L1/TEX と並ぶ希少資源になっている。

### 10.4 §6.3 / §7.4 の異常の説明

「命令数や wavefront を減らしたのに遅い」変更には、いずれも
**アドレス計算の追加**という共通点がある。`Lift_mat` を 12→8 ロードに
減らした版（§7.4、27% 悪化）も、ストアをコンフリクトフリーにする置換
（§5、596 µs）も、削ったメモリ側より増やした整数側のほうが大きかった、
と読める。**このカーネルでは「メモリ命令 1 本を減らすために整数命令を
2 本増やす」取引は成立しない。**

### 10.5 profiling 上の注意（重要）

ncu の単発 launch と実運用の launch は、**速度だけでなく順位も入れ替わる**。

| 版 | ncu duration | 実運用 1 launch 相当 |
|---|---:|---:|
| ベース | 433.8 µs | 284 µs |
| 版 B（メモリ重・整数軽） | **450.3 µs**（最遅） | 286 µs |
| swizzle + staging（整数重） | 445.7 µs | **300 µs**（最遅） |

ncu はリプレイのたびに L2 を流すので DRAM 側を重く見積もり（1 launch あたり
1.53×）、**メモリ律速に寄った条件でカーネルを測っている**。実運用では RK
ステージ間で `q,u,v,w` が L2 に残るため、相対的に発行律速へ寄る。
§7.3 は「この 2 つは別物として扱うこと」と書いたが、実際には
**最適化の優劣判定そのものが逆転しうる**。
`--set full` の Memory Workload Analysis だけを見て「L1/TEX 95% だから
メモリ律速」と判断すると、整数命令を増やす方向の変更を選んでしまう。
ncu で候補を絞った後は、必ず `nstep=1000` の end-to-end で確認すること。

### 10.6 結論と次の一手

- §9 の「XOR swizzle の範囲では解が無い可能性が高い」は**誤り**。解は 3072 個
  あり、実機でも予測どおりコンフリクトを消す。ただしそれでは勝てない。
- M 側 gather の shared 化は、L1/TEX 律速を外したうえでもなお負ける。
  **この案は最終的に不採用**（§9 の結論は変わらないが、理由が変わった）。
- 代わりに**新しい標的が見えた**: このカーネルの整数・アドレス演算。
  ベースの 1.227e9 整数 thread 命令（全体の 43%）を減らす方向の変更は、
  これまで一度も試されていない。ncu の Memory Workload Analysis を見ている
  限り視界に入らない標的である。

コードはベース（`e910708`）に戻してある。

---

## 11. 整数・アドレス演算の削減: §10.6 の標的を実測した

実施日: 2026-08-25。§10.6 が「これまで一度も試されていない」と書いた標的、
すなわち `tendency_fused_p7_tc_kernel` の整数・アドレス演算そのものを削る。
ベースは commit `326b80b`（カーネルは `e910708` のまま、device 0.852 秒）。
end-to-end は login node の `nstep=1000`、`CUDA_VISIBLE_DEVICES` で空いている
1 GPU に固定し、版を 1 回ずつ交互に回して 12 ラウンド。ncu は Slurm job
`44819`（`-s 5 -c 1`、1 launch、`nstep=10`）。条件は §5 と同一
（p=7, `NeX=NeY=NeZ=32`, `dt=1.0D-5`, `OPT1`）。

### 11.1 3 つの版

**版 I（整数演算だけを削る）**。数値も命令の種類も変えず、アドレスの
再計算だけを恒等式で置き換える。

- ペアで触る shared アクセスの 2 本目を、定数 XOR 1 本で導く。
  contraction の `k0 = 4` オペランドは `k0 = 0` のものと**その swizzle が
  読まないビット 1 本**しか違わない（x は bit 2、y は bit 5、z は bit 8）ので
  `fx ^ 4`, `fy ^ 32`, `fz ^ 256` でよい。アキュムレータ `c1` の格納先も
  `c0` の `^ 1` である。コンパイラはこの恒等式を証明できず、毎回
  swizzle を作り直していた。
- epilogue の面添字を `(i,j,k)` 経由ではなく `tid` から直に作る。
  `node1 = tid = i + 8j + 64k1` なので `tid >> 3` が `j + 8*k1`、
  `tid & 63` が `i + 8*j` であり、face 2/4 は `64 + (tid>>3)`、
  face 5/6 は `256 + (tid & 63)` と**シフト 1 本**で出る。

**版 II（x, y をレジスタに置く）**。x と y のアキュムレータは
**同じ 2 ノードを指す**（どちらも `C[row][2*colk (+1)]`, 平面 `k = warp`）ので、
shared を経由する必要がない。`sDx` / `sDy` を廃し、担当ノードを
アキュムレータの並び `(i,j,k) = (row, j0_c(+1), warp)` に合わせた。
shared store 4 本と load 4 本、および `__syncwarp()` 2 本が消える。

**版 III（版 II + 出力の転置）**。版 II の担当ノードは
`(lane>>2) + 16*(lane&3) + 64*warp` で、1 warp の足跡が 448 バイトに散った
64 バイト × 4 本になる。sector 数は連続アクセスと同じだが**キャッシュライン数が
2 → 4 に倍増する**。そこで x と y を転置して評価する:
`m8n8k4` では転置は**同じ 2 つのオペランド値を逆順に渡すだけ**（`A[j][m]` は
転置前の `B[m][j]` と同じ shared アドレス、`B[m][i]` は同じ fragment 要素）で、
命令もアドレス計算も増えない。これで thread `tid` の持つ 2 ノードが
`2*tid`, `2*tid + 1` になり、warp が 64 連続ノードを覆う。`Escale` と `dqdt`
は 16 バイトアクセス 1 本ずつになり、`sDz` と `sflux_bnd` と `sLift` も
隣接ペアとして読める。

### 11.2 命令数（SASS 静的、`cuobjdump -sass`）

| | 全命令 | 整数 | LDS | STS | LDG | regs |
|---|---:|---:|---:|---:|---:|---:|
| ベース | 380 | 176 | 36 | 16 | 44 | 32 |
| 版 I | 359 | 152 (−14%) | 36 | 15 | 44 | 32 |
| 版 II | 330 | 140 (−20%) | 26 | 11 | 44 | 32 |
| 版 III | 316 | **134 (−24%)** | **23** | **11** | 41 | 32 |

整数は `IMAD`+`LOP3`+`LEA`+`SHF`+`IADD3`+`ULEA`+`UIADD3`+`VIADD`。
どの版も spill 0、レジスタ 32 本、8 ブロック / SM を維持している
（`__launch_bounds__(256, 8)` の上限がちょうど 32 本なので、これは条件である）。

### 11.3 end-to-end（`nstep=1000`, 交互 12 ラウンド）

| 版 | CUDA device | 対ベース |
|---|---:|---:|
| ベース | 0.8518 | — |
| 版 I | 0.8509 | −0.1%（ノイズと同程度）|
| 版 II | 0.9506 | **+11.6%** |
| 版 III | **0.8488** | **−0.35%** |

版 I とベースの差は run 間のばらつき（同一版で 0.845〜0.854）に埋もれる。
版 III の −0.35% は 3 セットの独立な測定（10 ラウンド、5 ラウンド、
交互 12 ラウンド）で同符号・同程度に再現した。

### 11.4 ncu（job `44819`、ベース対 版 III、1 launch）

| | ベース | 版 III | 比 |
|---|---:|---:|---:|
| duration | 434.6 µs | 405.5 µs | **0.933** |
| ALU pipe 命令 | 2.464e7 | 1.416e7 | **0.574** |
| 全命令（warp） | 8.857e7 | 7.127e7 | 0.805 |
| shared load 命令 | 9.437e6 | 6.029e6 | 0.639 |
| shared store 命令 | 3.670e6 | 2.359e6 | 0.643 |
| shared store wavefronts | 1.366e7 | 7.624e6 | 0.558 |
| shared store バンクコンフリクト | 6.353e6 | 2.414e6 | **0.380** |
| shared load バンクコンフリクト | 2.235e5 | 2.456e5 | 1.099 |
| global load 命令 | 9.306e6 | 8.520e6 | 0.915 |
| global load sectors | 9.594e7 | 9.594e7 | **1.000** |
| L2 read sectors | 1.226e8 | 1.227e8 | 1.001 |
| DRAM read | 1.468 GB | 1.466 GB | 0.999 |
| L1/TEX throughput | 95.70% | **90.04%** | 0.941 |
| SM throughput | 32.36% | 26.79% | 0.828 |
| **long scoreboard stall** | 25.49 | **34.99** | **1.373** |
| mio throttle stall | 5.23 | 2.82 | 0.539 |
| barrier stall | 7.27 | 7.57 | 1.041 |
| achieved occupancy | 96.89% | 96.89% | 1.000 |

### 11.5 結論

- **§10.6 の見立ては外れである。** 整数演算だけを 14% 削った版 I は、
  end-to-end で測定できる変化を生まない。§10 で整数量が 4 版の順位を
  きれいに説明したのは相関であって、**整数を減らせば速くなるという関係では
  なかった**。発行スロットはこのカーネルの律速ではない。
- **本当に効いたのはキャッシュライン数である。** 版 II は版 III と
  ほぼ同じだけ命令を削っていながら 11.6% 遅い。両者の違いは epilogue の
  global アクセスの足跡だけで、**sector 数も DRAM 転送量も同じ**まま
  ライン数が 2 → 4 になっただけである。L1/TEX が 95% のカーネルでは
  この差が支配的になる。
- **版 III は単発 launch で −6.7%、実運用で −0.35%。** ALU −43%、
  shared 命令 −36%、store コンフリクト −62%、L1/TEX 95.7 → 90.0% と、
  減らしたかったものはすべて減っている。それでも実運用の利得が小さいのは、
  `long scoreboard` stall が 25.5 → 35.0 に**増えた**からである。
  仕事を削った結果、律速が発行スロットと L1/TEX から
  **global load レイテンシ**に移った。global の sector 数・L2・DRAM は
  1.000 倍で変わっていないので、これは「同じ量のロードを、より少ない
  他の仕事で隠さなければならなくなった」という意味である。
  8 ブロック / SM・occupancy 96.9% でもこのカーネルは global load を
  隠しきれていない。
- ncu と実運用の乖離は §10.5 と**逆向き**に出た。§10.5 では ncu が
  DRAM を重く見て整数の重い版を不当に速く見せたが、今回は ncu が
  −6.7%、実運用が −0.35% で ncu が利得を過大評価している。
  どちらにせよ **ncu の順位は end-to-end で確認するまで信じない**。

### 11.6 採用したもの、次の一手

版 III を採用した（版 I の恒等式も含む）。利得は小さいが、命令数・
shared traffic・バンクコンフリクトのいずれも単調に改善しており、
コードも `sDx` / `sDy` と `idx_dxy()` が消えて短くなる。

次の標的は **global load のレイテンシと sector 数**である。§9 / §10 で
「M 側 gather を shared に付け替える」案は不採用になっているが、
それは L1/TEX を減らす方向の話だった。今回分かったのは、残っているのが
帯域ではなく**依存の深さ**だということなので、筋としては

1. face gather（`VMapM` / `VMapP` を引いてから `q,u,v,w` を引く 2 段依存）を
   前倒しして、volume の load と重ねる。
2. `q,u,v,w` の 4 本を 1 本の構造体配列にする、あるいは `Escale` の 3 成分を
   16 バイト × 2 でまとめて、outstanding なロード数を減らさずに
   命令数と依存段数を減らす。

の 2 つが考えられる。いずれも数値契約（`AGENTS.md`）を変えない範囲で可能である。

---

## 12. face gather の前倒し: §11.6 の「次の一手」その 1 を実測した（不採用）

§11.6 は、残る律速が global load の**レイテンシ**（`long scoreboard` 25.5 → 35.0）
であることを受けて、筋を 2 つ挙げた。その 1 番目、

> face gather（`VMapM` / `VMapP` を引いてから `q,u,v,w` を引く 2 段依存）を
> 前倒しして、volume の load と重ねる。

を実装して測った。**結論は不採用**である。効かない理由は、§11 までの議論に
出てこなかった別の制約にある。

### 12.1 出発点: SASS 上の依存の並び

ベース（`4f37384`）の `tendency_fused_p7_tc_kernel` の SASS を見ると、
1 スレッドの前半は次の順に並んでいる。

```
0x2b0-0x3a0  LDG.E.64 x8      volume の q,u,v,w（node1, node2）
0x3c0-0x520  STS.64   x4      sFluxX/Y/Z への格納   ← ここで volume load を待つ
0x560        LDG.E    R21     VMapM[fidx]           ← 待った後で初めて発行
0x610-0x7e0  LDG.E.64 x8      q,u,v,w の iM / iP    ← さらにもう一度待つ
```

`VMapM` の 32 bit ロードが volume ロードの**後ろ**にあるため、1 スレッドは
global レイテンシを 2 回直列に踏む。ソース順を変えればこの 2 回を 1 回に
畳めるはず、というのが §11.6 の見立てだった。

### 12.2 2 つの版

- **版 A（index だけ前倒し）**: `iM` / `iP` の 2 本のロードだけを volume flux
  セクションの前に移す。face gather の残り（`q,u,v,w`、`normal_fn`、`Fscale`）は
  元の位置のまま。
- **版 B（セクションごと入れ替え）**: face gather セクション全体を volume flux
  セクションの前に置く。演算も格納先も変えない純粋な並べ替えで、どちらの
  セクションも `__syncthreads()` の前にあるため意味は不変。

いずれもレジスタ 32 本・spill 0・smem 16256 B で、ベースと同じ 8 ブロック/SM に
収まる。

### 12.3 測定結果

p=7 `CUDAFORTRAN_FUSED_TC`, `Ne = 32^3`, `nstep = 1000`, GB200 1 GPU、
`make CUDA=1`（`-gpu=ccnative`）。3 版を 1 ラウンドずつ交互に回した。
値は `CUDA device fused tendency`（秒）。

| ラウンド | ベース | 版 A | 版 B |
|---:|---:|---:|---:|
| 1 | 0.8464 | 0.8659 | 0.8484 |
| 2 | 0.8548 | 0.8688 | 0.8515 |
| 3 | 0.8491 | 0.8700 | 0.8505 |
| 4 | 0.9446 | 0.8671 | 0.8475 |
| 5 | 0.8515 | — | — |
| 代表値 | **0.850** | **0.868** | **0.849** |

ベースの 4 ラウンド目 0.9446 は外れ値で、他の 4 回は 0.846–0.855 に収まる。

- **版 A は +2.1%**（4 回とも 0.866–0.870 で、ばらつきの外）。
- **版 B は ±0**（−0.1% はノイズと同程度）。

### 12.4 なぜ効かないか

**版 A**: ソース上で先頭に出しても、ptxas が `LDG.E`（`VMapM`）を volume の
全ロードと数個の `STS` の後ろへ**押し戻す**。位置はベースの 0x560 に対し
版 A で 0x590 と、実質変わらない。`iM` / `iP` の生存区間が伸びた分だけ他の
スケジューリングが窮屈になり、正味で遅くなる。

**版 B**: 狙いどおり `VMapM` / `VMapP` が最初の data load になり（0x2a0 / 0x2d0）、
依存する `q[iM]` も 0x350 から出る。ところが今度は volume のロードが 0xa00
以降へ押し出され、**露出する依存チェーンが入れ替わっただけ**で総和が変わらない。

押し戻しと押し出しの原因は同じで、**このカーネルはレジスタ 32 本ちょうどで
余裕がゼロ**だからである。

```
ptxas info : Used 32 registers, 0 bytes spill stores, 0 bytes spill loads, 16256 bytes smem
```

256 スレッド × 8 ブロック/SM = 2048 スレッド、レジスタファイル 64K で
1 スレッド 32 本が上限である。§7 で得た 8 ブロック/SM を保つ限りこの枠は
動かせないので、**ロードを 1 本余分に飛ばしたまま保持する余地が構造的に無い**。
前倒しは、原理的にレジスタを消費する最適化である。

### 12.5 より本質的な理由

仮にレジスタに余裕があっても、この方向は効きにくい。achieved occupancy 96.9%、
64 warp/SM のこのカーネルでは、**1 スレッド内の memory-level parallelism を
増やしても、メモリ要求のパイプは他の 63 warp が既に埋めている**。

§11.4 で `long scoreboard` が 25.5 → 35.0 に増えたのを「global load レイテンシが
律速に移った」と書いたが、より正確には**「発行できるロードが足りない」のでは
なく「L1 で待たされている」**である。同じ表で global sector 数 1.000 倍、
L2 1.001 倍、DRAM 0.999 倍のまま L1/TEX が 90.0% に張り付いていたことと整合する。

したがって **§11.6 項目 1 の筋は誤り**で、残る余地は「レイテンシを隠す」側では
なく **「L1 トランザクション数そのものを減らす」側**にしかない。

### 12.6 その方向の在庫

L1 トランザクションを減らす案は、この調査ですでにほとんど試している。

| 案 | 節 | 結果 |
|---|---|---|
| M 側 gather を shared に移す | §9 | 不採用（global sectors −26% に対し shared バンクコンフリクト 20 倍、正味 3.7% 遅い）|
| contraction と face gather を同時にコンフリクトフリーにするレイアウト | §10 | 存在するが実機では遅い |
| `Lift_mat` 512×6 を分離可能 `Lift1D` 48 値に置換 | §7 | **採用**（global sectors 3684 → 2916）|
| x/y 導関数を shared を通さずレジスタ保持＋転置 epilogue | §11 | **採用**（キャッシュライン 4 → 2）|
| TMA で volume の `q,u,v` を smem へ直接運ぶ | `tma_survey.md` §3 | 不採用（2026-08-26）|

最後の行だけ後から足したので補足する。TMA は global→smem を **L1/TEX を
通さずに**運ぶので、この表で唯一「減らした分の代償が L1 に戻ってこない」案
だった（同一バイトの対照実験で global sector 1678 万 → 0、shared store
コンフリクト 128 万 → 0、レジスタ 30 → 16）。落ちたのは**レイアウト**である。
§10.2 の判定法を、TMA が 8 B 要素で符号化できる 4 通りのレイアウト全部に
機械的に当てると、**すべてで x/y 収縮が 2-way コンフリクト**になる。
TMA の swizzle は行ビットを必ず最下位のチャンクビットから当てるため
ノード bit4 がアドレス bit1 に落ち、n1 自身の像と衝突して階数が 3 に落ちる。
`sw_xy` が要る対応は bit4 → bit2 で、TMA の swizzle 語彙には無い。

なお `qM * VelM` は `sFluxX/Y/Z` の `iM` 成分から作れる
（`q[iM]*(u n1 + v n2 + w n3) = (qu)[iM] n1 + (qv)[iM] n2 + (qw)[iM] n3`）が、
`alpha = 0.5|VelP + VelM|` が `VelM` 単体を要求するため、
`u,v,w` の M 側ロードは消せない。これは §9 で測った案そのものでもある。

### 12.7 採用したもの

**無し。**作業ツリーは `4f37384` のまま。§11.6 の項目 1 はここで閉じる。
項目 2（`Escale` や `q,u,v,w` のロード幅拡大）は命令数を減らす案であって
L1 トランザクション数を減らす案ではないので、§12.5 の診断からすると
期待値は低い。

---

## 13. ±x 面の M 側 gather を shared 経由にした（採用、2026-08-26）

§12.7 は「§11.6 の項目 2 は L1 トランザクション数を減らす案ではないので
期待値は低い」で終わっていた。その判断自体は正しかったが、**どのロードが
L1 を食っているかを命令単位で測っていなかった**。測ったら、標的は
`q,u,v,w` のロード幅ではなく **6 面のうち 2 面だけ**だった。

対象 commit: `63a4234`。測定 GPU / ビルドは本レポート冒頭と同じ。

### 13.1 超過セクタの帰属（Slurm job `49589`）

`-lineinfo` 付きで `cuda_dg_kernels_tc.cu` を再ビルドし、
`ncu --set full --import-source yes` の Source ページを SASS 単位で見た。
`tendency_fused_p7_tc_kernel` の global ロード 95,944,704 セクタの内訳:

| ロード | 命令数 | セクタ | 理想 | 比 |
|---|---:|---:|---:|---:|
| face 点 0-255 の gather（`q,u,v,w` × M/P） | 8 | 41,551,872 | 16,777,216 | **2.48** |
| face 点 256-383 の gather（同上） | 8 | 8,388,608 | 8,388,608 | 1.00 |
| volume の `q,u,v,w` | 8 | 16,777,216 | 16,777,216 | 1.00 |
| `Escale` / `normal_fn` / `Fscale` / `VMap` ほか | 17 | 29,227,008 | 29,227,008 | 1.00 |

超過分は 24,774,656 セクタで、ncu が SOL で挙げる
「excessive sectors 24,772,608（全体の 25%）」と一致する。
**このカーネルの L1 の無駄は 1 か所に全部ある。**

理由は `Fmask` の並びである。face 点は面ごとに 64 点で、

| 面 | 法線 | ノード添字 | warp 32 レーンの足跡 | セクタ/warp |
|---|---|---|---|---:|
| 1, 3 | ∓y | `i + 8*j0 + 64*k` | 8 連続 × 4 本 | 8（理想）|
| **2, 4** | **±x** | `i0 + 8*j + 64*k` | **8 doubles 飛び** | **32** |
| 5, 6 | ∓z | `i + 8*j + 64*k0` | 32 連続 | 8（理想）|

±x 面だけ 1 レーン 1 セクタになり、32 B のうち 8 B しか使わない。
face 点 256-383（面 5, 6）に超過がゼロなのはこのためで、
測定はこの構造をそのまま映している。

### 13.2 対策

±x 面の **M 側**は、同じ要素の `i = 0` と `i = 7` のノードであり、
volume セクションが coalesce したロードで**すでにレジスタに持っている**。
そこで、その 2 面ぶんだけ shared にステージングして face セクションから読む。

- `sMface[4 * 144]`（4.6 KB）。field major、plane ストライド 72 doubles。
  face 点 32 個が連続 32 doubles を読むのでロードはコンフリクトフリー、
  plane を 72（≠ 16 の倍数）ずらしてあるので、1 ストア phase に同居する
  2 plane が同じバンクに落ちない。
- 書くのは `node & 7` が 0 か 7 のスレッドだけ。1 warp 32 レーンのうち 8 本。
- 読み側は `fp & 64` が面 2/4 を、`fp & 128` が plane を選ぶ。
  **warp 単位で分岐が揃う**ので発散しない。面 1,3,5,6 は従来どおり global。
- volume ストアと face 読み出しの間に `__syncthreads()` が 1 つ増える。

`VMapM` は面 2/4 では読まなくなる。P 側は隣接要素なので手が出ない。

### 13.3 測定（Slurm job `49612` / `49642`、ncu 1 launch）

| | ベース（`63a4234`） | 採用版 | 比 |
|---|---:|---:|---:|
| duration | 403.6 µs | **388.4 µs** | 0.962 |
| global load sectors | 95,944,704 | **78,643,200** | **0.820** |
| DRAM read | 1.469 GB | 1.454 GB | 0.989 |
| L1/TEX throughput | 90.32% | 90.97% | 1.007 |
| SM throughput | 26.88% | 31.73% | 1.180 |
| **long scoreboard stall** | 35.07 | **24.03** | **0.685** |
| barrier stall | 7.58 | 7.27 | 0.959 |
| mio throttle stall | 2.78 | 3.77 | 1.356 |
| shared store wavefronts | 7.62 M | 13.51 M | 1.772 |
| achieved occupancy | 96.92% | 96.60% | 0.997 |
| static shared / block | 16,256 B | 20,864 B | 1.283 |
| registers / thread | 32 | 32 | 1.000 |

`nstep=1000`（login node、版を交互に 4 ラウンド）:

| | ベース | 採用版 | |
|---|---:|---:|---:|
| Main（graph off） | 1.1092 | **1.0656** | −3.9% |
| `CUDA device fused tendency` | 0.8497 | **0.8060** | **−5.1%** |
| Main（graph on） | 1.0738 | **1.0367** | −3.5% |

数値は `SCALE_DG_VARYING_COEFF=1` で `63a4234` と**ビット一致**
（`Ne=2³`, `Ne=4³`、owned `dqdt` 全点）。`CUDAFORTRAN_SPLIT` との差は
1.8e-15 / 2.9e-14 で従来と同じ。

### 13.4 途中で外した 3 つの版と、その理由

| 版 | device 時間 | 判断 |
|---|---:|---|
| 上記 + `cudaFuncSetAttribute(..., cudaSharedmemCarveoutMaxShared)` | 1.114 s | **不採用**。shared を増やした結果 8 ブロック/SM を割ると思い込んで足したが、実際は既定のままで 9 ブロックぶん取れている。carveout を最大にすると L1 データキャッシュが削られ、**L1 が最繁ユニットのこのカーネルでは +31%** になる |
| `{q,u}` `{v,w}` の `double2` ペアでステージング | 0.829 s | 不採用。shared 命令は半分になるがレジスタが足りず 4 B スピルする |
| 面 2 だけステージング（shared 2 KB） | 0.819 s | 不採用。20 B スピル。shared を節約する意味は無かった |
| **field major・2 面（採用）** | **0.806 s** | 採用 |

最初の行が本節で一番高くついた誤りである。**shared を増やす前に
`launch__occupancy_limit_shared_mem` を測れば足りた**のに、机上で
「164 KB / 8 ブロック」を割って足りないと決めつけ、L1 を犠牲にする
属性を足していた。切り分けは、barrier だけを足した版（0.853 s = 変化なし）と
volume セクションだけ書き換えた版（0.845 s）を測って初めて確定した。

### 13.5 残っているもの

L1/TEX は 91% のままで、**律速は動いていない**。動いたのは long scoreboard
（35.1 → 24.0）で、§11.5 が「L1 を空けると global load レイテンシに移る」と
書いた向きの逆に進んだ。次に大きい単一項目は P 側 gather の超過
（残り約 1,240 万セクタ）だが、これは隣接要素のデータなのでブロック内に無い。

shared store のバンクコンフリクトが 2.41 M → 4.10 M に増えている。
ステージングの 8 命令はレイアウト上コンフリクトフリーのはずなので、
ここは未解明のまま残る。ただし L1/TEX が飽和していない今の版で
時間に効くかは別問題で、§8.9 の教訓（推定改善余地は経路単独の上限）が
そのまま当てはまる。

---

## 14. D1D フラグメントのレジスタ常駐化（不採用、2026-08-27）

`p31_gap_study.md` §14 で Nq=32 の TC カーネルが CUDA core 版の 2.66 倍になった
主因の 1 つは、**転置形にすると D1D 側のフラグメントが外側ループに依存せず、
レジスタに常駐できる**ことだった（mma 1 本あたり shared ロードが 2 本から 1 本になる）。
同じ手が p=7 に効くかを測った。**効かない。**

### 14.1 出発点 —— 6 本の mma が読む D1D はたった 2 double

`tendency_fused_p7_tc_kernel` の 6 本の mma は、すべて
`sDfrag[frag]` と `sDfrag[frag+32]` だけを読んでいる（`frag = (row<<2) + colk`）。
レジスタ 4 本で 6 本の shared ロードが消えるように見える。

### 14.2 ところが ptxas は既に半分やっていた

SASS（`cuobjdump -sass`、`-arch=sm_100`）を読むと、**x と y は既に 1 回のロードを
共有している**:

```
LDS.64 R14, [R28+UR8]        <- sDfrag[frag]        （z ブロック）
LDS.64 R10, [R28+UR8+0x100]  <- sDfrag[frag+32]
DMMA ; DMMA
   ... BAR.SYNC ; BAR.SYNC ...            <- sDz の往復
LDS.64 R20, [R28+UR8]        <- sDfrag[frag]        再ロード
LDS.64 R6,  [R28+UR8+0x100]  <- sDfrag[frag+32]     再ロード
DMMA(x) ; DMMA(y) ; DMMA(x) ; DMMA(y)     <- R20 と R6 が各 2 本に給餌
```

再ロードが起きるのは **`sDz` の往復が挟む 2 本のバリアを跨げない**からである。
したがって実際の余地は 6 本ではなく **2 本**（LDS 全 27 本のうち）である。

### 14.3 32 レジスタの天井に余白が無い

このカーネルは `__launch_bounds__(256, 8)` で 8 ブロック/SM を取っており、
その条件は 65536/256/8 = **レジスタ 32 本以下**である。ベースは**ちょうど 32 本**。

| 版 | レジスタ | スピル | LDS | LDL | STL | SASS 総命令 |
|---|---|---|---|---|---|---|
| ベース | 32 | 0 | 27 | 0 | 0 | 384 |
| 2 フラグメントをレジスタへ | 32 | **8 B** | 25（−2） | 1 | 1 | **384**（±0） |
| 1 フラグメントだけ | 32 | **8 B** | 26（−1） | 1 | 1 | **385**（+1） |

**ptxas は 32 本を守るためにローカルメモリへスピルする。** local も L1/TEX を通るので、
**shared 命令 2 本をローカル命令 2 本に置き換えただけ**になり、総命令数は動かない。
1 フラグメントだけにしても同じくスピルし、こちらは総命令が 1 本増える。

### 14.4 実測（`nstep=1000`、交互 6 ラウンド、占有 GPU、Slurm job `59470`）

| | Main [ms/step] | CUDA device [s] |
|---|---|---|
| ベース | **1.06510** | **0.802499** |
| 2 フラグメントをレジスタへ | 1.07178（**+0.63%**） | 0.809579（**+0.88%**） |

device 時間は **6 ラウンドすべてでベースが速い**。出力は
`input_p7_val_tc.conf`（`SCALE_DG_VARYING_COEFF=1`）で**ビット一致**なので、
差は純粋に実装コストである。**不採用。コードはベースに戻してある。**

### 14.5 なぜ移植できなかったか

Nq=32 で効いたのは、**平面掃きの構造では D1D フラグメントが j に依存せず、
かつ 512 スレッド / 126 レジスタという余白のある予算で回っていた**からである。
p=7 のカーネルは要素まるごとを shared に載せる別構造で、
`sDz` の往復がバリアを 2 本挟むため値をレジスタに保持する期間が長く、
しかも 8 ブロック/SM を成立させる 32 レジスタにちょうど張り付いている。
**買うための余白が無い。**

これは §11 が別方向から突き当たったのと同じ壁である。§10.4 は
「メモリ命令 1 本を減らすためにアドレス演算を 2 本増やす取引は成立しない」と
記録したが、本節はその一般化にあたる: **このカーネルでは、メモリ命令を
レジスタで買うこともできない。**

なお p=15 の TC カーネルは `launch_bounds(1024,1)`（上限 64 レジスタ）で、
同じ手には 12 double = 24 レジスタが要る。512 スレッド化とセットでなければ
入らず、shared は既に LSU の 41.1%（L1/TEX 67.9%、job `59436`）まで下がっている
ので、見返りは p=7 より薄い。**着手しない。**

## 15. 経路横断の再測定（2026-08-29）

p=7 専用の gap study は無いので、現行時間を本調査メモに置く。
[`README.md`](README.md) のまとめ表と同じ測定。commit `2dadc41`、login GPU 1、
3-run 中央値。入力は `conf_perf_p7.conf`（`Ne=32³`、`nstep=20`、graph off）の
`DqdtKernel_Type` だけを差し替え。µs/stage は `CUDA device *`（SPLIT は 4 本 +
elembnd、OpenACC は volume wall + elembnd）。

| 経路 | Main [ms/step] | µs/stage |
|---|---:|---:|
| `OPENACC_ASIS` | 3.492 | 1065.8 |
| `OPENACC_SPLIT` | 2.708 | 807.8 |
| `CUDAFORTRAN_SPLIT` | 2.565 | 764.1 |
| `CUDAFORTRAN_FUSED_DFMA` | 1.528 | 427.8 |
| **`CUDAFORTRAN_FUSED_TC`** | **1.073** | **274.9** |
| `CUDAFORTRAN_GEMM` | 5.088 | 1635.4 |

**最速は `CUDAFORTRAN_FUSED_TC` のまま。** 本文 §0 の ncu 時間は当時のカーネルのまま残す。
この日の namelist `CUDAFORTRAN_FUSED` は iso-schedule DFMA である。
旧 Fortran 融合（device 〜324 µs）は CC 最適の旧測であり、DFMA の 427.8 µs とは並べない。

**（追記 2026-09-02）本節の `FUSED_DFMA` 427.8 µs を p=7 の表の値とする。**
§17 に同じ量の別測定 **424.1 µs**（同日・同 conf・login 3-run 中央値）があり、
差は 0.87% で 3-run のばらつきの範囲だが、[`README.md`](README.md) の p=7 表の
他の行（`OPENACC_*`・`SPLIT`・`FUSED_TC`・`GEMM`）は**本節の同一セット**から
来ているので、行間の比較を別測定の混合にしないために本節の値を採る。
メカニズム比は 274.9 / 427.8 = 1.56×、§17 の分母では 1.57× で結論は同じ。

## 16. p=7 TC: z の shared 往復の天井（2026-08-29、不採用）

Fortran 融合がレジスタで z を完結していることの取り分を、p=15 §14.3 と同じ
不正アブレーションで測った。z の mma アキュムレータを生成したレーンで
`Escale_z` に掛け、`sDz` へのストア・ロードとそれを挟む `__syncthreads` 2 本を
消す。ノード写像は壊れる（数値は検証しない）。コードは測ったあと戻した。

| | commit / 入力 | GPU |
|---|---|---|
| ベース | `e3115fa`、`conf_perf_p7.conf`（`nstep=20`、graph off） | login GPU 1（`CUDA_VISIBLE_DEVICES=1`） |
| ビルド | `make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100` | nvhpc-hpcx/26.3 |

µs/stage = `CUDA device fused tendency` / (19 step × 3 RK)。3-run。

| 版 | Main [ms/step] | µs/stage |
|---|---:|---:|
| ベース（往復あり） | 1.0715 / 1.0702 / 1.0734 | 274.11 / **274.14** / 274.64 |
| z 往復を除去 | 1.0616 / 1.0598 / 1.0612 | 270.52 / **270.50** / 270.84 |

中央値は **274.14 → 270.50 µs（−3.6 µs、−1.3%）**。同一セッションのベース
再測定のレンジは 0.5 µs で、差はその数倍あるのでゼロではない。
p=15 の同アブレーションは +0.4%（誤差）だった。p=7 は L1/TEX が厚いので
shared 往復の代金がわずかに見える。

**天井が 4 µs なので実装しない。** 正しい写像に直すには z の mma 出力を
エピローグのレーンへ合わせるか、z だけ長さ 8 の FMA にする必要があり、
どちらも 32 レジスタ・8 ブロック/SM を崩す（§14）。そのリスクに対して
上限 1.3% では見合わない。

## 17. p=7 CUDA-core 融合の C++ 復活（2026-08-29）

`CUDAFORTRAN_FUSED` を Fortran `2dadc41^` の自然順・長さ 8 内積カーネルとして
C++ に戻した。login GPU 1、`conf_perf_p7.conf` の `DqdtKernel_Type` だけ
`CUDAFORTRAN_FUSED`、3-run 中央値。

| 経路 | Main [ms/step] | µs/stage |
|---|---:|---:|
| `CUDAFORTRAN_FUSED`（CC） | 1.227 | 326.8 |
| `CUDAFORTRAN_FUSED_DFMA` | 1.517 | 424.1 |
| `CUDAFORTRAN_FUSED_TC` | 1.074 | 274.9 |

CC 326.8 µs は旧 Fortran 〜324 µs と同水準。論文の主比は **TC / FUSED = 1.19×**、
メカニズム比 TC / DFMA は 1.54×。**（追記 2026-09-02）この表の `FUSED_DFMA` 424.1 µs は
§15 の 427.8 µs と同じ量の別測定である**（差 0.87%）。表の値としては §15 の
427.8 を採る（理由は §15 の追記）。ここの 424.1 は測定として残す。点変化係数、`Ne=2³`、`nstep=1` の owned `dqdt`
は `FUSED` と `FUSED_TC` がビット一致、`CUDAFORTRAN_SPLIT` との最大絶対差
1.78e-15。

**（追記 2026-08-30）** その後の CC 最適化は [`p7_gap_study.md`](p7_gap_study.md)。
device 326.3 → **302.8 µs/stage（−7.2%）**、主比 **1.10×**。
