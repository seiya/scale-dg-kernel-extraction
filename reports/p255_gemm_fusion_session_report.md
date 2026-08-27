# p=255 volume GEMM / fusion セッション報告

- 日付: 2026-08-25
- 対象: SCALE-DG 3D advection kernel extraction（`CUDAFORTRAN_GEMM` / `CUTE` / `FUSED`）
- 条件（性能比較）: PolyOrder=255、Ne=1、nstep=1000、入力は `bench_runs/p255_gemm*.conf`。時間はホスト wall ではなく **CUDA Event の device-event**。

数値契約は変更していない。`q,u,v,w,Escale` は点ごと、volume 項は `D(q*vel)`、6面の数値フラックス、halo は維持。

---

## 1. 用語

| 名前 | 意味 |
|---|---|
| device-event | tendency 全体の GPU 時間（境界フラックス、volume flux、volume GEMM、lift、assembly または fused epilogue） |
| volume GEMM | x/y/z の体積微分 GEMM だけ。FUSED では z の assembly epilogue も含む |
| cuBLAS / GEMM | `CUDAFORTRAN_GEMM`。volume 微分は cuBLAS |
| CUTE | `CUDAFORTRAN_GEMM_CUTE`。パイプラインは GEMM と同じで、volume の 3 GEMM だけ CUTLASS d884 Tensor Core |
| FUSED（最終形） | `CUDAFORTRAN_GEMM_FUSED`。flux は materialize。z GEMM の epilogue で assembly |

---

## 2. 実験 A/B/C: `q*vel` を GEMM operand に融合するか

当初の仮説は、`volume_flux_kernel` を消すために CUTLASS mainloop で `q*vel` を生成することだった（`MulPairIterator` 等）。

同じ nstep=1000 での結果（セッション前半）:

| | パス | device-event |
|---|---|---|
| A | volume_flux + **cuBLAS** + assembly | **約 3.87 s** |
| B | volume_flux + **CUTLASS d884** + assembly | **約 3.91 s**（volume GEMM のみ約 2.28 s） |
| C | GEMM mainloop で `q*vel` を生成 | **約 6.48 s** |

A と B の比は約 1.01（全体で約 1%）。CUTLASS 化そのものの差は小さい。

C は B の約 **1.66 倍**（+約 2.56 s）。消した `volume_flux` は以前の nsys から 1000 step 換算でおおむね **0.45–0.5 s** 程度。融合のために **2.5 s 以上余分に払っていた**。

解釈:

- 標準 mainloop の global→shared（`cp.async` 的な経路）を、dual `ld.global` + FP64 multiply が壊す。
- materialized flux なら L2 再利用できるタイルを、CTA ごとに `q` と `vel` から再生成する。
- したがって p=255 では **volume flux を GEMM に fuse しない**。flux 配列は「無駄な中間」ではなく、高効率 dense GEMM のための前処理。

方針:

```text
volume_flux → CUTLASS/cuBLAS GEMM → assembly
```

operand fusion は数値的には GEMM と丸め誤差で一致しうるが、性能上は捨てた。

---

## 3. 次の実験: assembly だけ z GEMM epilogue に載せる

flux 融合をやめたあと、FUSED は一時的に CUTE と同じ構造（assembly は別カーネル）に戻していた。その時点では時間が CUTE と揃うのは当然。

実装した最終パイプライン:

```text
volume_flux
→ CUTLASS x GEMM → deriv_x
→ CUTLASS y GEMM → deriv_y
→ lift（cuBLAS、従来どおり）
→ CUTLASS z GEMM（d884 mainloop はそのまま）
     + epilogue: dqdt = -(Ex*Dx + Ey*Dy + Ez*Dz + lift)
```

lift は z epilogue が読むため、**z GEMM より前**に移した。`dqdt_assembly_kernel` は FUSED では起動しない。x/y GEMM は CUTE と同一タイル。

検証: `SCALE_DG_VARYING_COEFF=1` で owned `dqdt` を `CUDAFORTRAN_GEMM` と比較。maxabs は **3.55e-15**（丸め）。

性能（本セッション後半の実測）:

| | device-event | volume GEMM |
|---|---|---|
| cuBLAS | **3.881 s** | — |
| CUTE | **3.914 s** | 2.284 s |
| FUSED（assembly epilogue） | **3.603 s** | 2.444 s |

CUTE より約 **0.31 s（約 8%）**、cuBLAS より約 **7%** 速い。z epilogue の追加ロードで volume GEMM は **2.284 → 2.444 s**（+0.16 s）だが、別カーネル assembly を消した方が得。

---

## 4. epilogue 微修正の試行（いずれも不採用）

基準: FUSED device-event **3.60–3.64 s**、volume GEMM **約 2.445 s**。速くなったものだけ残す約束で順に入れた。

### 4.1 barrier 削減（acc を smem にまとめて載せる）

iteration ごとの `__syncthreads()` 2 回を、全 fragment を smem に書いてから 2 回にする案。

- 数値: GEMM との maxabs **約 500**（不正）
- 時間: 改善なし（約 3.61 s）

CUTLASS 標準 epilogue は **同じ smem スロットを iteration ごとに再利用**する。warp iterator を進めてまとめて書くとタイル対応が壊れた。**不採用（コードは元に戻した）。**

### 4.2 auxiliary fragment の寿命短縮

`Ez*Dz` のあと `Ex*Dx`、`Ey*Dy`、lift を段階加算。barrier の外で 6 fragment を同時に持たない。

- 数値: maxabs **約 2e-15**（可）
- 時間: **約 3.70 s**（volume GEMM 約 2.54 s）→ 約 0.09 s 悪化

直列ロードのレイテンシの方が、レジスタ圧の緩和より大きい。**不採用。**

### 4.3 標準 epilogue に近いフルアンロール

`#pragma unroll(1)` を `kIterations` フルアンロールに変更（標準 CUTLASS の light functor 側）。

- 数値: maxabs **約 3.6e-15**（可）
- 時間: **約 3.69 s**（volume GEMM 約 2.53 s）→ 悪化

**不採用。** 残しているのは 2 sync / iteration、6 operand をまとめて読む、`unroll(1)`。

---

## 5. いまのコード配置

| ファイル | 役割 |
|---|---|
| `cuda_cutlass_gemm_fused.cu` | CUTE の 3 GEMM、FUSED の x/y GEMM と z assembly 起動 |
| `cutlass_z_gemm_assembly.h` | z batched GEMM の TensorOp mainloop + 自前 assembly epilogue |
| `mod_cuda_dg_kernels.cuf` | FUSED: flux → xy GEMM → lift → z assembly。CUTE は従来の assembly カーネル |
| `mod_advect3d_eq.f90` | `CUDAFORTRAN_GEMM_CUTE` / `GEMM_FUSED` の dispatch と作業配列 |
| `Makefile` | `CUTLASS_HOME`（既定 `third_party/cutlass`） |

CUTLASS 本体とジョブ/ダンプはリポジトリに含めていない。ビルドは NVIDIA HPC SDK の module のうえ `make CUDA=1`（GB200 では `GPUFLAGS=-gpu=cc100`、nvcc は `sm_100`）。

関連コミット: `299a868`（`feature/cuda`）。

---

## 6. セッションの結論

1. **p=255 で flux を GEMM operand に融合してはいけない。** cuBLAS と CUTLASS d884 はほぼ同じ。遅い主因は fusion による mainloop 破壊。
2. **flux の materialization は有効な前処理**である。
3. **assembly の z-epilogue 融合は有効**（本環境で CUTE 比約 8%）。mainloop は触らない。
4. その epilogue に対する barrier まとめ・fragment 分割・フルアンロールは、このタイルでは効かないか悪化する。

次に手を付けるなら、同じ方針で **lift と z の重なり**や **assembly 以外の独立カーネル**であり、`q*vel` の mainloop 融合はやり直さない。

---

## p=255 Tensor Core カーネルのタイル化（2026-08-27）

コミット: 本節を追加したコミット（親は `f9d4dfb`）。GPU は RIKYU GB200 1 枚、
`make CUDA=1 GPUFLAGS=-gpu=cc100`、nvcc `-arch=sm_100`。時間は Slurm job
`59500` / `59538` / `59571` / `59586` / `59592` / `59602`、ncu は `59500`（旧）と
`59571`（新）。入力は `conf_perf_p255*.conf`（`Ne=1`、`nstep=20`、graph off）。
測定はすべて `sbatch` で占有した GPU 上のもの。

### 1. 旧カーネルの何が悪かったか

`tendency_{x,y,z}_p255_tc_kernel` は **1 ブロック 32 スレッド（1 warp）**で
8×8 の出力タイル 1 枚だけを持ち、K=256 の全 64 ステップで **mma の両オペランドを
global から読み直していた**。1 mma あたり 96 double を動かして 512 FLOP なので
**0.67 FLOP/B**、1 stage あたり約 39 GB が L1 を通る。

ncu（job `59500`、`-s 6 -c 8`）:

| カーネル | dur [µs] | SM% | DRAM% | **L1/TEX%** | warps% | L2 hit% | L2 [GB] |
|---|---|---|---|---|---|---|---|
| `tendency_z_p255_tc_kernel` | 2576.6 | 17.9 | 44.2 | **98.1** | 49.5 | 27.4 | 20.07 |
| `tendency_y_p255_tc_kernel` | 2553.6 | 18.1 | 3.3 | **99.2** | 49.6 | 91.8 | 12.40 |
| `tendency_x_p255_tc_kernel` | 2533.8 | 18.2 | 2.6 | **99.1** | 49.5 | 92.5 | 11.31 |

**3 本とも L1/TEX 98〜99% に張り付き、SM は 18% しか出ていない。**
`p31_gap_study.md` §13.5 が Nq=32 で見たのと同じ律速だが、原因が違う。
あちらはオペランドが既に shared にあり、答えは**レジスタ常駐化**だった。
ここはオペランドが global から来るので、答えは **shared へのタイル化**である。

### 2. 変更

3 本を **1 本のテンプレートカーネル**（`DIR = 0,1,2`）に統合した。

- **64×64 の出力タイルを 256 スレッド（8 warp）で持つ。** 各オペランドを
  1 列ごとではなく 64 列に 1 回読むので、オペランド転送量が **8 分の 1** になる。
- **3 方向すべてを転置形 `C^T[m][n] = Σ_l A[m][l]·B[n][l]` で書いた。**
  両オペランドが同じ `[outer][l]` の形になるので、shared レイアウトもローダも
  1 種類で済む。転置は m8n8k4 ではオペランドの順序でしかなく無料であり、
  **エピローグが coalesce する**（旧版は 1 レーンが 256 または 65536 離れた
  2 ノードを持っていた）。
- **1 warp が持つ 8 タイルを 1×8 ではなく 2×4 に並べた。** アキュムレータは
  どちらも 8 対だが、k-step あたりのオペランドロードが 1+8=9 本から
  2+4=6 本に減る。**同じレジスタで shared ロードが 3 分の 2 になる。**
- shared レイアウトは `l + 16*outer` の 1 種類、swizzle も 1 本:

```c
sw255(idx) = idx ^ (((idx >> 4) & 3) << 2) ^ (((idx >> 6) & 3) << 0)
```

  この配列には 3 種類のアクセスが来る —— mma 読み（`l` が `lane%4`、
  `outer` が `lane/4`）、outer 高速のストア（D1D と y/z のフラックス）、
  `l` 高速のストア（x のフラックス）。bit 4-5 を bit 2-3 に畳むと読みが、
  bit 6-7 を bit 0-1 に畳むと outer 高速ストアが直り、**それぞれの畳み先が
  他方では不変なので 1 本で 3 つとも conflict free になる**（実装前に
  全 phase を列挙して確認した）。

面の分担は旧版どおり x が face 2,4、y が face 1,3、z が face 5,6 である。
z は線形 (i,j) 添字で縮約するので平面ループを持たず、grid は 3 方向とも
`4096*Ne` で揃う。レジスタ 64、shared 16 KB、スピル 0、4 ブロック/SM。

### 3. 結果

| 版 | Main [ms/step] | tendency [µs/stage] | FP64 ピーク比 |
|---|---|---|---|
| 旧（1 warp/block） | 13.4943 | 4474.3 | 14.6% |
| 64×64 タイル、warp 1×8 | 5.9592 | 1931.4 | 33.7% |
| **64×64 タイル、warp 2×4（採用）** | **4.8927** | **1566** | **41.6%** |
| （参考）`CUDAFORTRAN_GEMM_FUSED` | 3.2105 | 999.1 | 65.2% |

**旧版の 2.86 倍。** 1×8 → 2×4 の register blocking だけで 1.237 倍。

ncu（job `59571`、3 インスタンス平均）: L1/TEX **99% → 54.2%**、
SM **18% → 64.6%**、DRAM 12.2%、L2 転送量 11〜20 GB → 2.37 GB/launch。
**律速は L1/TEX から演算側に移った。**

**それでも `GEMM_FUSED` には 1.57 倍負ける。** p=255 は演算律速の次数で、
`GEMM_FUSED` は CUTLASS の GEMM に epilogue 融合を載せてピークの 65.2% を
出している。ここを手書きで抜くのは大きな FP64 GEMM で CUTLASS を上回る作業であり、
**本節はそこまで行っていない。p=255 の最速経路は `CUDAFORTRAN_GEMM_FUSED` のままである。**

### 4. 採用しなかったもの —— global ロードのソフトウェアパイプライン化

残る余地は占有率とレイテンシに見えた（warps active 46.9%、**issue active 31.7%**）ので、
`p31_gap_study.md` §14 で効いた「次のチャンクを mma の前に発行する」手を当てた。

| 版 | レジスタ | スピル | µs/stage |
|---|---|---|---|
| パイプラインなし（採用） | 64 | 0 | **1563.5〜1569.1** |
| パイプライン、`launch_bounds(256,3)`、マクロ記述 | 80 | 48 B | **1519.9〜1522.8**（−3.0%） |
| 同、`launch_bounds(256,2)` | 124 | 0 | 1644.7（**+4.8%**） |
| 同、ラムダ記述 | 80 | 68 B | 1566.4（**±0**） |
| 同、ロードを 2 か所に展開して記述 | 80 | 68 B | 1571.7（**+0.5%**） |

**−3.0% はマクロで書いたときだけ出る。** 意味的に同じ 2 通りの書き換え
（ラムダ、手展開）では 68 B スピルになって効果が消える。
**レジスタ割り当ての当たりくじであって設計ではない**ので採用しない。
§10.3 / §10.4（`tc_paper_survey_2407.09621.md`）が記録した
「静的指標が改善しても時間が悪化する」型の罠と同じ側にある。

### 5. 検証

`SCALE_DG_VARYING_COEFF=1` で `SCALE_DG_DUMP_DQDT` により完全な `dqdt(:,1:Ne)` を
落とし、`CUDAFORTRAN_GEMM`（`input_p255_val_gemm.conf`）と全点比較した。

| メッシュ | 最大絶対差 | 相対差 | >1e-14 | >1e-13 |
|---|---|---|---|---|
| `Ne=1` | 3.553e-15 | **4.156e-16** | 0 | 0 |
| `Ne=2`（AGENTS.md の Ne>1 スモーク） | 3.553e-15 | **4.156e-16** | 0 | 0 |

タイル化と 2×4 register blocking は**互いにビット一致**する（総和順序を変えないため）。
4.156e-16 は `GEMM_FUSED` が他次数で記録している値と同じである。

**既存次数の回帰**: 変更前バイナリと 10 ケースすべて**ビット一致**
（p=7 の `FUSED_TC` / `FUSED`、p=15 の `FUSED_TC` / `FUSED`、p=31 の
`FUSED_TC` / `FUSED` / `GEMM`、p=255 の `FUSED` / `GEMM` / `GEMM_FUSED`）。
非 CUDA ビルドと OpenACC ビルドも通る。

### 6. 残っているもの

- **BM=BN=128 にするとオペランド転送量がさらに半分**になる。
  512 スレッドか 16 タイル/warp が要り、レジスタと占有率の再調整になる。
  ただし L1/TEX は既に 54% で律速から外れているので、効くとしても
  演算側（SM 64.6%）を通してである。
- **この構造は Nq に依存しない。** `NQ255` を引数にすれば p=63 / p=127 にも
  そのまま載る。両次数には融合経路が CUDA core 版すら無く、
  `GEMM_FUSED` もピーク比 28.4% / 46.7% と p=255 の 65.2% よりずっと低いので、
  **勝てる見込みは p=255 より高い。**
