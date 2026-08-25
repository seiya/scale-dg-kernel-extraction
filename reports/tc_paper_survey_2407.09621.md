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
GB200 の SM 数・クロックを入れて `execution_times.md` 系に一欄追加するのが
最も安価な改善。

### B-1. mma 形状の拡大【論文の先を行く部分】

論文は A100 のため `m8n8k4` しか選択肢がないが、sm_90 以降は FP64 の
`m16n8k4 / m16n8k8 / m16n8k16` が使える。k を深くすれば同じ演算量あたりの
shared 読み出し回数と命令数が減り、A-1 と同じく L1 律速の緩和に効く。
GB200 = sm_100 での利用可否の確認が必要。

**重要な前提差**: 論文の 2.3× のうち 2× 分は「A100 では FP64 Tensor Core peak が
FP64 CUDA-core peak の 2 倍」という事実に由来する。**GB200（Blackwell）では
この 2× アドバンテージが撤廃されており、FP64 Tensor Core peak = FP64 CUDA-core
peak = 40.1 TFLOP/s である**（`overall_summary_report.md` §7、
`gpu_optimization_session_report.md` §7 に記録）。したがって本リポジトリで
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
（本リポジトリの `CublasEmulation` フラグ。現環境では API 不在）と同じ発想。
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
  `execution_times.md` の「p=7 では FUSED が最速」「FUSED_TC は 1.28× 遅い」という
  記載は、この変更で覆った。同ファイルは commit `299a868` 時点の測定なので未更新。
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
- §2 B-1（`m16n8k*` FP64 mma）は未着手。ただし `Compute (SM)` は 32% で、
  演算パイプは余っている。優先度は低い。
- §6.3 の異常（wavefront 減 = 遅くなる）と §7.4 最終行の異常は、
  どちらも「命令数・wavefront 数を減らしたのに遅い」形をしている。
  マイクロベンチで shared ストアと発行スケジューリングを単独測定するのが早い。
