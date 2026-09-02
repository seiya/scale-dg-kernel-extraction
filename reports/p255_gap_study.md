# p=255 (Nq=256, Ne=1) — GEMM 融合から手書き Tensor Core で CUTLASS を抜くまで

測定環境: RIKYU NVIDIA GB200 1 GPU、`make CUDA=1 GPUFLAGS=-gpu=cc100`（nvcc は
`-arch=native` = `sm_100`）。コミット: 本レポートを追加したコミット。
入力: `conf_perf_p255_tc.conf` / `conf_perf_p255.conf`
（`NeX=NeY=NeZ=1, PolyOrder=255, dt=1e-7, nstep=20, UseCudaGraph=.false.`）。
ncu 用は `conf_perf_p255_tc_ncu.conf`（`nstep=4`）。
ncu は Slurm job `62367`（改修前）、`62642`（二重バッファ導入時点）、
`63045`/`63050`（最終形）。いずれも `--metrics` 指定の CSV。
検証: `input_p255_val_tc.conf` / `input_p255_val_gemm.conf` と、それらの
`NeX=2` 版、`SCALE_DG_VARYING_COEFF=1`。

**時間の読み方**: §1 以降の µs/stage は実行ファイルが出す
`CUDA device fused tendency`（または経路をまたぐ比較では両方が出す
`Volume derivate + surface lift`）を `WarmupStep` を除いた 19 ステップ ×
3 RK ステージ = 57 で割った値である。§0 の GEMM 融合セッションは
`nstep=1000` の device-event（秒）で、当時の測定条件のまま残す。

**測定環境の注意**: ログインノードには GB200 が 4 枚あり、他ジョブと同居すると
同一バイナリの測定が 7% 揺れることがあった。本レポートの数値はすべて
`CUDA_VISIBLE_DEVICES=1` に固定し、3〜5 回の中央値を採っている。
これを固定せずに採った測定は一度捨てている。

## 0. 前史: volume GEMM 融合と Tensor Core タイル化

旧 `p255_gemm_fusion_session_report.md`（2026-08-25 の GEMM / z-epilogue
融合と、2026-08-27 の TC タイル化）。数値は当時の測定のまま残す。
§1 以降がその結論（当時は `GEMM_FUSED` が最速）を覆す記録である。

- 日付: 2026-08-25
- 対象: SCALE-DG 3D advection kernel extraction（`CUDAFORTRAN_GEMM` / `CUTE` / `FUSED`）
- 条件（性能比較）: PolyOrder=255、Ne=1、nstep=1000、入力は `bench_runs/p255_gemm*.conf`。時間はホスト wall ではなく **CUDA Event の device-event**。

数値契約は変更していない。`q,u,v,w,Escale` は点ごと、volume 項は `D(q*vel)`、6面の数値フラックス、halo は維持。

---

### 0.1 用語

| 名前 | 意味 |
|---|---|
| device-event | tendency 全体の GPU 時間（境界フラックス、volume flux、volume GEMM、lift、assembly または fused epilogue） |
| volume GEMM | x/y/z の体積微分 GEMM だけ。FUSED では z の assembly epilogue も含む |
| cuBLAS / GEMM | `CUDAFORTRAN_GEMM`。volume 微分は cuBLAS |
| CUTE | `CUDAFORTRAN_GEMM_CUTE`。パイプラインは GEMM と同じで、volume の 3 GEMM だけ CUTLASS d884 Tensor Core |
| FUSED（最終形） | `CUDAFORTRAN_GEMM_FUSED`。flux は materialize。z GEMM の epilogue で assembly |

---

### 0.2 実験 A/B/C: `q*vel` を GEMM operand に融合するか

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

### 0.3 次の実験: assembly だけ z GEMM epilogue に載せる

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

### 0.4 epilogue 微修正の試行（いずれも不採用）

基準: FUSED device-event **3.60–3.64 s**、volume GEMM **約 2.445 s**。速くなったものだけ残す約束で順に入れた。

#### 4.1 barrier 削減（acc を smem にまとめて載せる）

iteration ごとの `__syncthreads()` 2 回を、全 fragment を smem に書いてから 2 回にする案。

- 数値: GEMM との maxabs **約 500**（不正）
- 時間: 改善なし（約 3.61 s）

CUTLASS 標準 epilogue は **同じ smem スロットを iteration ごとに再利用**する。warp iterator を進めてまとめて書くとタイル対応が壊れた。**不採用（コードは元に戻した）。**

#### 4.2 auxiliary fragment の寿命短縮

`Ez*Dz` のあと `Ex*Dx`、`Ey*Dy`、lift を段階加算。barrier の外で 6 fragment を同時に持たない。

- 数値: maxabs **約 2e-15**（可）
- 時間: **約 3.70 s**（volume GEMM 約 2.54 s）→ 約 0.09 s 悪化

直列ロードのレイテンシの方が、レジスタ圧の緩和より大きい。**不採用。**

#### 4.3 標準 epilogue に近いフルアンロール

`#pragma unroll(1)` を `kIterations` フルアンロールに変更（標準 CUTLASS の light functor 側）。

- 数値: maxabs **約 3.6e-15**（可）
- 時間: **約 3.69 s**（volume GEMM 約 2.53 s）→ 悪化

**不採用。** 残しているのは 2 sync / iteration、6 operand をまとめて読む、`unroll(1)`。

---

### 0.5 いまのコード配置

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

### 0.6 セッションの結論

1. **p=255 で flux を GEMM operand に融合してはいけない。** cuBLAS と CUTLASS d884 はほぼ同じ。遅い主因は fusion による mainloop 破壊。
2. **flux の materialization は有効な前処理**である。
3. **assembly の z-epilogue 融合は有効**（本環境で CUTE 比約 8%）。mainloop は触らない。
4. その epilogue に対する barrier まとめ・fragment 分割・フルアンロールは、このタイルでは効かないか悪化する。

次に手を付けるなら、同じ方針で **lift と z の重なり**や **assembly 以外の独立カーネル**であり、`q*vel` の mainloop 融合はやり直さない。

---

### 0.7 Tensor Core カーネルのタイル化（2026-08-27）

コミット: 本節を追加したコミット（親は `f9d4dfb`）。GPU は RIKYU GB200 1 枚、
`make CUDA=1 GPUFLAGS=-gpu=cc100`、nvcc `-arch=sm_100`。時間は Slurm job
`59500` / `59538` / `59571` / `59586` / `59592` / `59602`、ncu は `59500`（旧）と
`59571`（新）。入力は `conf_perf_p255*.conf`（`Ne=1`、`nstep=20`、graph off）。
測定はすべて `sbatch` で占有した GPU 上のもの。

#### 1. 旧カーネルの何が悪かったか

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

#### 2. 変更

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

#### 3. 結果

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

#### 4. 採用しなかったもの —— global ロードのソフトウェアパイプライン化

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

#### 5. 検証

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

#### 6. 残っているもの

- **BM=BN=128 にするとオペランド転送量がさらに半分**になる。
  512 スレッドか 16 タイル/warp が要り、レジスタと占有率の再調整になる。
  ただし L1/TEX は既に 54% で律速から外れているので、効くとしても
  演算側（SM 64.6%）を通してである。
**（訂正・2026-08-28）** 本節の結論「p=255 の最速経路は `CUDAFORTRAN_GEMM_FUSED`
のままである」は覆った。同じ 64×64 タイルのまま、チャンクループの二重バッファ化と
1 ワープ 4×4 mma タイル、staging の double2 化、エピローグの組み替えで
`FUSED_TC` は 1563.9 → 968.8 µs/stage になり、`GEMM_FUSED` を 1.035 倍上回る。
**p=255 の最速は `CUDAFORTRAN_FUSED_TC` である。** 上の表と本節の記述は
当時の測定としてそのまま残す。詳細は本レポート §1 以降。
また §4 の「global ロードのソフトウェアパイプライン化は当たりくじであって設計ではない」
という判断も、そこで試したのがレジスタプリフェッチだけで **shared の二重バッファ化を
伴っていなかった**ためである。二重バッファにすると 27.7% の再現する利得になる。

- **この構造は Nq に依存しない。** `NQ255` を引数にすれば p=63 / p=127 にも
  そのまま載る。両次数には融合経路が CUDA core 版すら無く、
  `GEMM_FUSED` もピーク比 28.4% / 46.7% と p=255 の 65.2% よりずっと低いので、
  **勝てる見込みは p=255 より高い。**

## 1. 出発点

§0.7 のタイル化（2026-08-27）で `FUSED_TC` は 64×64 タイル・1 ブロック 256 スレッド・
1 ワープ 2×4 の mma タイル・単一バッファの形になり、そこで
**「`GEMM_FUSED` には 1.57 倍負ける。p=255 の最速経路は `CUDAFORTRAN_GEMM_FUSED`
のままである」**と結論していた。本レポートはその結論を覆す。

改修前の実測（同一条件で採り直したもの）:

| 経路 | Main [ms/step] | device 融合 tendency [µs/stage] |
|---|---:|---:|
| `CUDAFORTRAN_FUSED_TC`（改修前） | 4.8817 | 1563.9 |

## 2. 律速の確定（改修前）

ncu job `62367`、x カーネル（`DIR=0`）:

| 指標 | 値 |
|---|---:|
| duration | 648.1 µs |
| SM throughput | 64.4% |
| L1/TEX throughput | 52.6% |
| DRAM throughput | 10.1% |
| warps active | 47.8%（4 ブロック/SM × 256 スレッド） |
| issue active | 32.7% |

stall 内訳（`warps_issue_stalled_*_per_issue_active`）:

| long_sb | math_pipe | wait | short_sb | barrier | mio |
|---:|---:|---:|---:|---:|---:|
| **7.09** | **6.90** | 2.85 | 2.31 | 2.14 | 0.07 |

**どの資源も飽和していない**。最大の stall は global ロード待ち
（`long_scoreboard`）で、これはチャンクループが

```
global ロード → shared ストア → barrier → mma → barrier
```

とプリフェッチ無しで並んでいることの直接の結果である。

アブレーション（いずれも数値としては誤り、賞金の天井を測るためだけのもの）:

| 消したもの | µs/stage | 対 baseline |
|---|---:|---:|
| baseline | 1565.1 | — |
| チャンク 0 以外の staging を全部 | 1368.3 | **−12.6%** |
| mma を素の FMA に（FLOP −81%） | 1392.1 | **−11.1%** |
| barrier 2 本とも | 1443.0 | **−7.8%** |

**単独で支配的な項目は無い。**

## 3. 形状スイープ（単一バッファのまま）

カーネルを `<DIR, BM, BN, TM, TN, THREADS, MINB>` でテンプレート化して
実測した（`TM×TN` は 1 ワープが持つ 8×8 mma タイル数）。
mma ループは `TM*TN` 回の mma に対して `TM+TN` 本の shared オペランドロードを
払うので、比は 1×8 で 1.125、2×4 で 0.75、4×4 で 0.5 になる。

| 形状 | µs/stage | 対 baseline | レジスタ |
|---|---:|---:|---:|
| 64×64, 2×4, 256 thr（当時の採用形） | **1565.1** | — | 64 |
| 64×64, 1×8, 256 thr | 1705.9 | +9.0% | 64 |
| 64×64, 4×4, 128 thr | 1938.4 | +23.9% | 128 |
| 128×128, 2×4, 1024 thr | 1698.2 | +8.5% | 64 |
| 128×64 / 64×128 / 128×128 の 4×4 各種 | 1840〜2250 | +18〜44% | 128〜254 |

**オペランドロードを減らす方向はすべて負ける。** 4×4 は mma ループの
ロード比を 3 分の 2 にするのに 23.8% 遅い。理由は蓄積器で、4×4 は
32 double = 64 レジスタを占め、128 レジスタ・占有率 25% になる。

同じ 4×4 形状に §2 のアブレーション（staging 全消し）を当てると
**mma ループだけなら 4×4 の方が 10.6% 速い**（1225.1 対 1368.3 µs/stage）
一方で staging の代金が 12.5% から **30%** に膨らんでいた。
**mma ループの勝ちを staging がそっくり食っている**、というのがこの時点の診断である。

### 3.1 スウィズル計算のハイスト（このときは不採用）

SASS を見るとチャンクループ本体 244 命令のうち **126 命令（52%）が整数の
アドレス計算**だった。`sw255` は index の下位 4 bit しか触らないので、
`l = 4*ks + colk`、`outer = 8*t + row` に対して

```
sw255(l + 16*outer) = 16*outer + (colk ^ c) + ((4*ks) ^ ((row & 3) << 2)),
c = (2*t + (row >> 2)) & 3
```

と、オペランドごとの項と k ステップごとの項に分かれる（colk が bit 0-1、
4*ks が bit 2-3、16*outer がそれ以上と、3 つの場が重ならないため）。
両方ループ不変なのでループ外に出せる。

結果は**ループ本体 244 → 202 命令（−17%）で 5.1% 遅い**（1565.1 → 1644.7）。
レジスタもスピルも同じ（64、0）である。
**この時点でこのカーネルは命令発行律速ではない**ことが確定した。
（後述のとおり、二重バッファ導入後は同じ変更が **1.7% 速く**なる。）

## 4. 効いた変更

### 4.1 チャンクループの二重バッファ化（−27.7%）

shared を 2 面持ち、

```
issue(k+1) → mma(buf) → store(buf^1) → barrier
```

とする。バリアはチャンクあたり 2 本から 1 本になり、次チャンクの global
ロードは mma ループ全体にわたって飛んでいる。
ロードした値はレジスタに生のまま置き、`q*vel` の積は**ストア時**に行う
（issue 時に掛けると、まさに待たせたくない場所でロードを待つことになる）。

これで **4×4 形状が使えるようになる**。両者は単独では効かず、組でしか効かない:

| 形状 | 単一バッファ | 二重バッファ |
|---|---:|---:|
| 64×64, 2×4, 256 thr | 1565.1 | 1334.5 (MINB=3) |
| 64×64, 4×4, 128 thr | 1938.4 | **1130.3 (MINB=3)** |

`__launch_bounds__` の第 2 引数（SM あたり最小ブロック数）はここでは
**当たりくじではなく、スピルしない最大の占有率**である:

| MINB | レジスタ | スピル | µs/stage |
|---:|---:|---:|---:|
| 2 | 204〜212 | 0 | 1548.0 |
| **3** | **166〜168** | **0** | **1130.3** |
| 4 | 128 | 64 B | 1353.6 |
| 5 | 96 | 392 B | 2056.1 |
| 6 | 80 | 648 B | 2389.4 |

3 ブロック × 128 スレッド × 168 レジスタ = 64512 で、65536 のレジスタファイルを
ほぼ使い切る。占有率は 18.75% だが、**パイプラインがある以上それで良い**という
のがこの節の内容である。

二重バッファ化した 128×128 は静的 shared の 48 KB 制限に当たり、
64×128 / 128×64 は入るが、最終形に対して 1.10〜1.36 倍と負ける。

### 4.2 staging の double2 化（−3.7%）

global で最速に走る添字方向に隣り合う 2 要素を 1 スレッドが持ち、
`double2` 1 本で運ぶ。ロード命令が半分になり、アドレスも半分しか作らない。
x のフラックス面（l が最速）は shared 側でも隣接するので `STS.128` になる。
1088.5 µs/stage、ビット一致。

### 4.3 mma ループのスウィズル・ハイスト（−1.7%）

§3.1 でそのまま実装したものを、パイプライン化後にもう一度当てた。
今度は **1088.5 → 1069.5 µs/stage で採用**。
**同じ変更が単一バッファでは 5.1% の損、二重バッファでは 1.7% の得**という
のがこの調査で最も分かりやすい「命令数は律速ではない」の証拠である。

### 4.4 エピローグの入れ替え（−8.6%）

エピローグをアブレーション（積の書き出しだけにする）で測ると
**カーネルの 17.5%** を占めていた。SASS では 1 ワープ 16 出力ペアに対して
**112 本の LDG**（`DIR=0`）が出ている。

原因は `TM*TN` タイルを平坦に回していたことで、読むもののうち半分は
両方の添字に依存しない: x では 2 つの面フラックスが m（タイル行）だけ、
4 つの lift 係数が n（タイル列）だけで決まり、y と z では逆である。
平坦に回すとコンパイラはタイルごとに読み直していた。

b を外、a を内にして、m 側をレジスタに退避、n 側を内側ループの外に出す。
lift 係数はレーンが持つノード対が n で隣接するので `double2` で入る。

| DIR | LDG（前） | LDG（後） |
|---|---:|---:|
| 0 | 112 | **32** |
| 1 | 96 | **48** |
| 2 | 96 | **48** |

977.3 µs/stage、ビット一致。残るエピローグは**アブレーションで 9.7%**、
その中身は `Escale` と `dqdt` の read-modify-write で、
どちらも数値コントラクトが要求する通信量そのものである。

なお、この変更は単一バッファの旧カーネルに当てても効く（1565.1 → 1175.3）。
本レポートの寄与分解は「二重バッファ → double2 → ハイスト → エピローグ」の
順で測った値である。

### 4.5 ペア格納順の入れ替えでストア側バンクコンフリクトを消す（−0.8%）

4.2 の副作用として、ncu が **shared ストアの 2-way コンフリクト**を検出した
（`st` コンフリクト 8.45 M、`st` wavefront 16.8 M ＝ 下限の 2 倍）。
outer が最速の面では 1 レーンが持つ 2 要素が `o` と `o+1` で、
スウィズルは `c(o+1) = c(o) ^ 4` を満たす。全レーンが偶数側を先に書くと、
半ワープ 16 レーンが 8 バンクにしか届かない。

**各半ワープの上位 8 レーンだけ奇数側を先に書く**と、2 本のストア命令が
それぞれ 16 バンクを覆う。実測でコンフリクトは 8.45 M → **70 K**、
`st` wavefront は 16.85 M → **8.46 M**（コンフリクトフリーの下限）。
968.8 µs/stage、ビット一致。

## 5. 最終形

| 段階 | µs/stage | 前段比 | 対 baseline |
|---|---:|---:|---:|
| 改修前（64×64, 2×4, 256 thr, 単一バッファ） | 1563.9 | — | — |
| + 二重バッファ & 64×64, 4×4, 128 thr, MINB=3 | 1130.5 | −27.7% | −27.7% |
| + staging の double2 化 | 1088.5 | −3.7% | −30.4% |
| + mma ループのスウィズル・ハイスト | 1069.5 | −1.7% | −31.6% |
| + エピローグ b 外 / a 内 & lift の double2 | 977.3 | −8.6% | −37.5% |
| + ペア格納順の入れ替え | **968.8** | −0.8% | **−38.1%** |

**すべての段階が改修前とビット一致**である（総和順序を変えていない）。

### 5.1 経路比較（同一条件、両者が出す `Volume derivate + surface lift`）

| 経路 | Main [ms/step] | µs/stage | FP64 ピーク比 |
|---|---:|---:|---:|
| **`CUDAFORTRAN_FUSED_TC`** | **3.1265** | **1014.2** | **63.4%** |
| `CUDAFORTRAN_GEMM_FUSED` | 3.2346 | 1050.5 | 61.2% |
| `CUDAFORTRAN_GEMM` | 3.4657 | 1124.2 | 57.2% |
| `CUDAFORTRAN_FUSED_TC`（改修前） | 4.8817 | 1588.9 | 40.5% |

理論演算量は体積 GEMM の `3 * 2 * Nq^4` = **2.577e10 FLOP/stage**（lift と
`Escale` の乗算は含まない）、ピークは GB200 の FP64 = 40.1 TFLOP/s
（`AGENTS.md` のとおり CUDA core と Tensor Core で同じ）。
device 側の `CUDA device fused tendency`（968.8 µs/stage）で採ると **66.3%**。

**p=255 の最速経路は `CUDAFORTRAN_FUSED_TC` になった**（`GEMM_FUSED` の 1.035 倍）。
§0.7 の「最速は `GEMM_FUSED` のまま」という結論はここで覆る。手書きの 64×64 タイル GEMM が CUTLASS の
`d884gemm_64x128_16x3` を上回れた理由は演算そのものではなく、
**エピローグを融合できることと、面 lift を同じカーネルで払えること**である。

## 6. 最終形の律速

ncu job `63050`、x カーネル:

| 指標 | 改修前 | 最終形 |
|---|---:|---:|
| duration | 648.1 µs | **557.8 µs** |
| SM throughput | 64.4% | **74.8%** |
| L1/TEX throughput | 52.6% | 49.4% |
| DRAM throughput | 10.1% | 11.7% |
| warps active | 47.8% | 18.1% |
| issue active | 32.7% | 28.2% |
| shared ld 命令 | 12.58 M | **8.39 M** |
| shared st コンフリクト | 45 K | 70 K |
| global sector / request | 55.0 M / 3.67 M（15.0） | 55.0 M / 3.67 M（15.0） |

stall 内訳:

| | long_sb | math_pipe | wait | short_sb | barrier |
|---|---:|---:|---:|---:|---:|
| 改修前 | 7.09 | 6.90 | 2.85 | 2.31 | 2.14 |
| 最終形 | **1.43** | **3.21** | **2.90** | 0.91 | 0.28 |

**global ロード待ちは 7.09 → 1.43 に落ち、バリアは 2.14 → 0.28 になった。**
残る最大は `math_pipe_throttle` + `wait` で、合計すると stall の約 3 分の 2 を
占める。global ロードは 1 リクエストあたり 15.0 セクタで完全に coalesce しており、
shared はロード・ストアともコンフリクトフリーの wavefront 下限に張り付いている。

最終形でのアブレーション:

| 消したもの | 対 最終形 |
|---|---:|
| staging 全部 | −18.6% |
| エピローグ（積の書き出しだけに） | −9.7% |
| mma → 素の FMA（FLOP −81%） | −2.8% |
| barrier | −2.7% |

**FLOP を 81% 落としても 2.8% しか速くならない**一方で、
`math_pipe_throttle` が最大の stall である。これは矛盾ではなく、
**どの資源も単独では律速していない**ことを言っている。改修前と同じ性質のまま、
全体が 1.61 倍速くなった。

## 7. 不採用にしたもの

| 候補 | 結果 | 機構 |
|---|---:|---|
| バリアをループ先頭へ（p=63 / p=127 で勝った手） | +10.7% | 16 チャンクでは節約は 1 本、代わりに毎回分岐が増える |
| staging アドレスのループ外ハイスト | +2.5% | ループ本体の命令 −34% でも時間は増える。命令数は律速でない |
| 128×128 / 128×64 / 64×128 タイル（全 thread 数・全 launch_bounds） | +10〜35% | レジスタと占有率で負ける。減るオペランド転送量では取り返せない |
| 1 ワープ 4×8 / 8×4 タイル | +65〜145% | 蓄積器 64 double でスピル |
| 単一バッファでのスウィズル・ハイスト | +5.1% | §3.1。パイプライン後は逆に −1.7% |
| `BK=32`（バリア半減） | 未実装 | プリフェッチのレジスタが 48 → 96 になり MINB=3 に入らない。静的 shared 48 KB も超える |
| D1D 面の `cp.async` | 範囲外 | shared の宛先がスウィズルで連続にならないので `cp.async` では書けない |
| x と y の融合（同じ出力タイルなので `dqdt` の RMW が 1 回減る） | 未実装 | 蓄積器とプリフェッチが 2 組必要で 224 レジスタを超える |

## 8. 検証

`SCALE_DG_VARYING_COEFF=1` で `SCALE_DG_DUMP_DQDT` により `dqdt(:,1:Ne)` を
全点ダンプして比較した。

| 比較 | 結果 |
|---|---|
| 最終形 対 改修前 `FUSED_TC`（`Ne=1`） | **ビット一致** |
| 最終形 対 `CUDAFORTRAN_GEMM`（`Ne=1`） | 最大絶対差 3.553e-15、相対 4.405e-16、>1e-14 が 0 件 |
| 最終形 対 `CUDAFORTRAN_GEMM`（`Ne=2`） | 最大絶対差 3.553e-15、相対 4.405e-16、>1e-14 が 0 件 |

既存次数の回帰（改修前バイナリとの比較、`SCALE_DG_VARYING_COEFF=1`）:
p=7 `FUSED_TC`、p=15 `FUSED_TC`、p=31 `FUSED_TC`、p=63 `FUSED_TC`、
p=127 `FUSED_TC`、p=255 `FUSED`（CUDA core 版）の 6 ケースすべて**ビット一致**。

## 9. 残っているもの

- **staging 18.6%**: global は完全 coalesce、shared はコンフリクトフリー、
  アドレスのハイストは逆効果。残るのは転送量そのもので、
  ブロックあたり `q`/`vel` を 4 回読む冗長度はタイルを大きくしないと減らず、
  タイルを大きくするとレジスタで負ける（§7）。
- **エピローグ 9.7%**: 中身は `Escale` と `dqdt` の read-modify-write で、
  数値コントラクトが要求する通信量。x と y の融合で `dqdt` の RMW を 1 回
  減らせるが、レジスタが足りない（§7）。
- **占有率 18.75%**: レジスタ 168 で決まっている。4 ブロック（128 レジスタ）は
  スピルして 20% 遅い。蓄積器 64 レジスタとプリフェッチ 48 レジスタが下限で、
  どちらも削ると別のところで負ける。
- **この構造は Nq に依存しない。** p=63 / p=127 の `FUSED_TC` は別設計
  （面フラックス先出し + xz/y 分割）で、そちらは 1024 スレッド・占有率 50%・
  バリア律速（`p127_gap_study.md` §11.11）という別の場所にいる。
  ここで効いた二重バッファとエピローグの入れ替えが向こうでも効くかは未測定である。

> **（追記 2026-08-30、§16）** 二重バッファ後の 64×128 タイルは +26%、`BK=32`
> 単一バッファは +37%、`elembnd` を消す天井は −2.44% だが重ねは +0.44%。
> 契約内で staging / 占有率を動かす手は測って閉じた。

---

## 10. `CUDAFORTRAN_GEMM_FUSED` 側の最適化（2026-08-28）

コミット: 本節を追加したコミット（親は `ba53906`）。GPU は RIKYU GB200 1 枚、
`CUDA_VISIBLE_DEVICES=1` 固定、`make CUDA=1 GPUFLAGS=-gpu=cc100`、nvcc は
`-arch=native` = `sm_100`。入力は `conf_perf_p255.conf` / `conf_perf_p255_tc.conf`
（`Ne=1`, `nstep=20`, graph off）、ncu 用は `conf_perf_p255_ncu.conf`（`nstep=4`）。
Slurm job は nsys `63119` / `64240`、ncu `63119` / `63994` / `64240`。
時間はすべて両経路が出す `Volume derivate + surface lift` を
`WarmupStep` を除いた 19 ステップ × 3 RK ステージ = 57 で割った値、5 回の中央値。

§5.1 の時点で p=255 の最速は `FUSED_TC`（1014.2）で、`GEMM_FUSED` は 1050.5
µs/stage と **1.035 倍**の差しかなかった。差が小さいので `GEMM_FUSED` 側を
詰めれば最速が入れ替わりうる、というのが本節の出発点である。

### 10.1 出発点と内訳

| | µs/stage |
|---|---:|
| `CUDAFORTRAN_GEMM_FUSED`（`ba53906`） | 1048.8 |
| `CUDAFORTRAN_FUSED_TC`（同） | 1014.0 |

nsys job `63119` のカーネル内訳（60 launch の中央値）:

| kernel | µs |
|---|---:|
| z GEMM + assembly (`GemmBatchedDqdtAssembly`) | 339.1 |
| x GEMM (`Gemm<64,128,16>`) | 254.3 |
| y GEMM (`GemmBatched<64,64,16>`) | 242.7 |
| `volume_flux_kernel` | 128.1 |
| `elembnd_flux_kernel`（side stream、x GEMM の中に収まる） | 26.7 |

体積 GEMM 1 方向の mma 下限は `2*Nq^4 / 40.1 TFLOP/s` = **214.2 µs**。
x は 84%、y は 88% の効率で回っており、**z だけが 63%** である。差の 124.9 µs が
assembly epilogue の代金で、経路全体の 11.9% にあたる。

### 10.2 律速の確定 —— z epilogue は命令発行律速である

epilogue のアブレーション（数値としては誤り、天井を測るためだけのもの）:

| 消したもの | µs/stage | 対 baseline |
|---|---:|---:|
| baseline | 1049.9 | — |
| `Escale_x`, `Escale_y` の読み（体積テンソル 5 → 3 本） | 1030.0 | **−1.9%** |
| `Dx`,`Dy`,`Ex`,`Ey`（5 → 1 本） | 1106.6 | **+5.4%** |
| lift 再構成をまるごと | 997.3 | **−5.0%** |
| epilogue 全部（`-Dz` を書くだけ） | 965.4 | **−8.0%** |

**ロードを 5 → 1 本に減らすと 5.4% 遅くなる。** オペランド数はこの
カーネルの律速ではない。lift の 4 本の面 gather を消しても、`Lift1D` の
読み 2 本を消しても、戻る時間は**どちらも同じ 990 µs**で加算的でなかった。

ncu job `63994` が答えを出した。lift を消した版と baseline の比較:

| | baseline | lift 無し |
|---|---:|---:|
| duration | 559.1 µs | **487.7 µs**（−12.8%） |
| `smsp__inst_executed.sum` | 125.4 M | **109.0 M**（−13.0%） |
| SM throughput | 75.7% | 84.7% |
| warps active | 12.26% | 12.21% |
| レジスタ | 254 | 184 |
| long_scoreboard / math_pipe / wait | 0.45 / 1.51 / 2.07 | 0.50 / 1.42 / 2.10 |

**時間が命令数にそのまま比例し、stall 内訳も占有率も動かない。**
レジスタが 254 → 184 に落ちても占有率が変わらないのは、ブロック数が
レジスタ（4）と shared 48 KB（4）の**両方**で 4 に制限されているためである。
したがってこの epilogue で効くのは命令数を減らす変更だけであり、
以下の 3 つはすべてその形をしている。

### 10.3 効いた変更

いずれも `Nq > 64` の枝（x GEMM が CUTLASS のとき）だけに適用する。
`Nq <= 64` の枝は x GEMM が cuBLAS で epilogue を持たないため (1) が成立せず、
(2) と (3) を単独で入れると p=63 が **+2.7% / +0.8%** と遅くなった（ビット一致で、
ptxas のコード生成が変わるだけ）。3 つは 1 つのテンプレート引数 `kWeighted` に
まとめてある。

#### (1) `Escale_x` / `Escale_y` の乗算を x / y GEMM の epilogue へ前送り（−1.0%）

z epilogue は 1 出力要素あたり `Dx`, `Dy`, `Escale_x`, `Escale_y`, `Escale_z` の
**5 本**の体積テンソルを読んでいた。x GEMM と y GEMM の epilogue で
`Escale` を掛けてしまえば 3 本になる。総転送量は変わらない（`Escale_x` も
`Escale_y` も 1 回ずつ読む）が、**402 MB が z から x/y へ移る**。

要点は**手書きの epilogue を書かないこと**である。CUTLASS 標準 epilogue の
source 経路に載る出力オペレータ `PointwiseScale`（`D = acc * C`、C は `Escale`）
として書けば、標準 epilogue の速い経路がそのまま使える。
手書き epilogue を y GEMM に入れた版も試したが、**それだけで +72 µs** かかり、
移動で得られる分の 4 倍を失う（§10.4）。

1048.8 → 1038.6 µs/stage。nsys で機構どおり: z 339.1 → **317.6**、
x 254.3 → 257.7、y 242.7 → 249.4 µs。

#### (2) 添字のクランプをタイル原点へ寄せる（−0.85%）

epilogue は行と列のすべての添字に `min(..., Nq-1)` / `min(..., nq2-1)` を
掛けていた。これをタイル原点に 1 回だけ掛ける形に変える。はみ出した行や列は
「本来とは違うが有効な」アドレスを読むだけで、値は出力イテレータが
ストアを述語化するので決して書かれない。これで**面 gather のアドレスが
行オフセットについてアフィンになり、強度低減が効く**。

1038.6 → 1030.1 µs/stage、ビット一致。

#### (3) lift の 6 本のロードを 3 本の 16 バイトロードにする（−0.75%）

lift は 1 出力要素あたり 6 本のロードを出す —— 面 2/4 を同じ添字 `j+kNq` で、
面 1/3 を同じ添字 `i+kNq` で、`Lift1D` の面 5/6 係数を同じ添字 `k` で。
**どのペアも添字を共有している**ので、ペアを隣接させれば `double2` 1 本で読める。

- `elembnd_flux_kernel` に `pair_nq2` を足し、面平面を (1,3), (2,4), (5,6) の
  3 組にインターリーブして書く。書く**バイト数は同じ**で、宛先添字が変わるだけ。
  このカーネルは side stream で x GEMM の裏に隠れている。
- `Lift1D` の第 5・6 列を並べた `dg_lift_zpair`（`2*Nq` 要素）を初回に 1 度だけ作る。
  `Lift1D` はメッシュ演算子で走行中に変わらない。

1030.1 → 1021.2 µs/stage、ビット一致。

### 10.4 不採用にしたもの

| 候補 | 結果 | 機構 |
|---|---:|---|
| `Ex*Dx + Ey*Dy` を y GEMM の epilogue で合成し z を 2 テンソルに | +8.1% | 手書き epilogue が CUTLASS 標準を置き換える代金だけで **+72 µs**。z 側の取り分は測ると **0**（stock y のまま z だけ 2 テンソルにすると +72.8 µs） |
| z epilogue を 5 → 2 / 5 → 1 テンソル（アブレーション） | +6.9% / +5.4% | 同上。オペランド数は律速ではない |
| z タイル 64×64 / 128×32、warp 32×16 / 16×32、stage 3 / 5、epilogue padding 4 / 16 | +0〜+5% | 現行 64×32・warp 32×32・4 stage・padding 8 が最良。改修後に測り直しても順位は同じ |
| x GEMM を y と同じ batched 形（A=D1D stride 0）に | +0.7% | 同じ演算・同じ k 順序・同じメモリ配置でもタイルの切り方で負ける |
| x GEMM を cuBLAS に（`Nq<=64` の規則を全次数へ） | +0.7% | `p63_gap_study.md` の「Nq=256 では wash」は現在は小さな負けである |
| x GEMM タイル 64×64 / 128×128 / 4 stage | +0.6〜+51% | 現行 64×128・3 stage が最良 |
| epilogue の面ポインタに `__restrict__` | ±0 | ストアとのエイリアスではない |
| lift の計算位置を shared staging の後ろへ | ±0 | ライブレンジでもない |
| (2)(3) を `Nq<=64` の枝にも適用 | +2.7% / +0.8% | ビット一致のまま ptxas のコード生成が悪化する。枝ごとに分けた |

### 10.5 最終形

| 段階 | µs/stage | 前段比 | 対 baseline |
|---|---:|---:|---:|
| `ba53906` | 1048.8 | — | — |
| + `Escale` を x/y epilogue へ | 1038.6 | −1.0% | −1.0% |
| + クランプをタイル原点へ | 1030.1 | −0.8% | −1.8% |
| + lift のロードを `double2` 化 | **1021.2** | −0.9% | **−2.6%** |

| 経路 | µs/stage | FP64 ピーク比 |
|---|---:|---:|
| `CUDAFORTRAN_FUSED_TC` | **1015.0** | 63.4% |
| `CUDAFORTRAN_GEMM_FUSED`（本節） | 1021.2 | 63.0% |
| `CUDAFORTRAN_GEMM_FUSED`（`ba53906`） | 1048.8 | 61.4% |

**p=255 の最速は `CUDAFORTRAN_FUSED_TC` のままである**（1.006 倍）。
差は 1.035 倍から **1.006 倍**に縮んだ。

nsys job `64240` の内訳: z **299.6**（339.1 から −11.6%）、x 258.0、y 249.8、
`volume_flux` 128.5、`elembnd` 27.7 µs。
ncu job `64240`: z は SM 79.6%（73.4 から）、命令 117.6 M（136.4 M から −13.8%）、
レジスタ 246、L1/TEX 59.0%、DRAM 15.5%、占有率 12.2%。

### 10.6 残っているもの

改修後のアブレーション（同一条件）:

| 消したもの | µs/stage | 対 baseline |
|---|---:|---:|
| baseline | 1021.3 | — |
| lift 再構成 | 989.9 | −3.1%（改修前は −4.8%） |
| epilogue 全部 | 975.3 | −4.5%（改修前は −6.1%） |

- **lift 31.4 µs**: 1 要素あたり 3 本の 16 バイトロードで、この因数分解の下限である
  （6 個の double を運ぶのに `LDG.128` は 3 本要る）。4 本の横面値は 1 ブロック内で
  再利用が無い（列タイル 64 は 1 つの `j` に収まるので `fb0[i+kNq]` はブロック内で
  全部相異なる）。体積サイズの lift 中間配列に戻す案は 2026-08-25 に測って遅い。
  §10.8 に、この行を閉じるために追加で潰した 2 本の道を書く。
- **epilogue の残り 14.6 µs**: `Dx`, `Dy`, `Escale_z` の 3 本と `dqdt` のストアで、
  どれも数値コントラクトが要求する通信量そのものである。
- **`volume_flux_kernel` 128.5 µs**: 893 MB を 7.09 TB/s（参照ピークの 90%）で
  流しており、materialize する限り下がらない。方向で割って side stream に
  載せる案は `overall_summary_report.md` §8.7 で不成立が測られている。
- **x / y / z の mainloop**: それぞれピークの 84 / 86 / 84%。タイル掃引は
  §10.4 のとおり全滅である。
- **カーネル間の隙間 14 µs**（nsys 実測、`volume_flux`→x 4.2、x→y 2.9、y→z 6.5 µs）。
  カーネルを減らす方向は §10.4 の x batched 化で負けている。

### 10.7 検証

`SCALE_DG_VARYING_COEFF=1` で `SCALE_DG_DUMP_DQDT` により `dqdt(:,1:Ne)` を
全点ダンプして比較した。

| 比較 | 結果 |
|---|---|
| p=255 `GEMM_FUSED` 対 `CUDAFORTRAN_GEMM`（`Ne=1`） | 最大絶対差 3.553e-15、相対 4.156e-16、>1e-14 が 0 件 |
| p=255 `GEMM_FUSED` 対 `CUDAFORTRAN_GEMM`（`Ne=2`） | 同上 |
| p=127 `GEMM_FUSED` 対 `CUDAFORTRAN_GEMM`（`Ne=2³`） | 最大絶対差 1.776e-15、相対 2.078e-16、>1e-14 が 0 件 |
| p=63 `GEMM_FUSED`（`Ne=4³`）対 改修前 | **ビット一致** |
| p=255 `GEMM_CUTE` / `FUSED_TC` / `FUSED` 対 改修前 | **ビット一致** |

4.156e-16 と 2.078e-16 は `GEMM_FUSED` が改修前から記録している値と同じである。
CUDA ビルドと非 CUDA ビルドの両方を通した。

### 10.8 残りを閉じにいった第 2 ラウンド（2026-08-28、採用ゼロ）

§10.6 の「残っているもの」のうち、安く測れるものを潰した。すべて負の結果である。

#### lift を shared にステージする案 —— 賞金ゼロ（実装せず）

面の値をエポローグ開始時に協調ロードで shared に置き、そこから読む案。
このカーネルは §10.2 のとおり**命令発行律速**なので、`LDG` が `LDS` に替わっても
命令数は 1 対 1 で変わらず、賞金は原理的にゼロである。アドレス計算も §10.3 (2) の
クランプ前送りで既にアフィン化されており、強度低減が効いている。

#### lift を x / y GEMM 側へ移す案 —— 数え上げで閉じた

エピローグの座標写像（行 ↔ n、列 ↔ m）で 1 出力要素あたりの lift ロード本数を
数えると、**z が 3 本、y が 3 本、x が 5 本**である。z が 3 方向のうち最良の宿主で、
移す先が無い。

「K を 2 本伸ばして lift 項を GEMM 本体に載せる」手も検討した。x なら
`A' = [D1D | lx1 | lx2]`（`Nq × (Nq+2)`）、`B' = [flux_x ; fb1 ; fb3]` の rank-2 拡張で
`Dx + lx` が出る。y も同様に組める（`A'` の追加 2 列が面 1/3、`B'` の追加 2 行が
`Lift1D` の第 1・3 列で、後者はバッチ間で共有できる）。**しかしその GEMM の
エピローグは `Escale` 乗算に使えなくなる**: 出るのは `Ex*(Dx+lx)` で、欲しいのは
`Ex*Dx + lx` である。`lx/Ex` は点ごとなので rank-2 に書けない。
x / y の K 拡張の有無 4 通りすべてで **z のロードは 6 本のまま**で、
交換の中身は「L2 常駐の面 gather 2 本」↔「DRAM の体積テンソル 2 本」になる。
前者は §10.3 (3) の実測から 1 本あたり 3.3 µs、後者は §10.3 (1) の実測から 1 本
あたり約 5 µs 相当なので、**差し引き +3.4 µs で悪化する**。
`EpilogueWithBroadcast` / `EpilogueWithVisitor` を使えば手書きエピローグの
+72 µs を避けられるが、上のとおり機構が無料でもロード数が増えるので意味が無い。

#### エピローグ反復ループの展開

| 展開 | µs/stage | 対 baseline |
|---|---:|---:|
| `#pragma unroll(1)`（採用） | **1021.5** | — |
| 全展開（`acc2smem` の実行時 switch が定数畳み込みされる） | 1031.8 | **+1.0%** |
| `#pragma unroll(2)` | 1022.2 | ±0 |

全展開は `acc2smem::push` の 4 分岐を消して命令を減らすが、レジスタ配置が悪化して
遅くなる。§0.4.3 が 6 オペランド版で見たのと
同じ向きで、エピローグを軽くした後でも結論は変わらない。

#### 占有率を上げる方向

z カーネルはブロック数がレジスタ（246）と shared（48 KB）の**両方**で 4 に
制限され、占有率 12.5% である。`cutlass::Kernel` は
`__launch_bounds__(kThreadCount, 1)` を固定で持つので、第 2 引数を出せる自前
ランチャを書いて振った。

| 版 | µs/stage | 対 baseline |
|---|---:|---:|
| stage 4, MINB=1（採用形と同じ意味） | 1030.7 | **+0.9%** |
| stage 3（shared 36 KB）, MINB=1 | 1029.7 | +0.8% |
| stage 3, MINB=5 | 1163.9〜1446.1 | **+14〜40%** |
| stage 4, MINB=4 | 1270.8 | **+24%** |

shared を 36 KB に落とせば 5 ブロックぶんの余地はできるが、レジスタ 246 が
4 で頭打ちにする。MINB を上げて無理に 4 / 5 ブロックにするとスピルして壊滅する。
§4.1 の「`__launch_bounds__` の第 2 引数は当たりくじではなく、スピルしない最大の
占有率」がここでも成り立つ。**意味的に同一の launch bounds を持つ自前ランチャに
差し替えるだけで +0.9%** というのは、本セッション 3 度目のコード生成の当たり外れ
（他の 2 つは §10.4 の (2)(3) を `Nq<=64` 枝に当てた場合）であり、当然採用しない。

#### 結論

**残るのは 3 本の GEMM の mainloop だけ**である（x 84%、y 86%、z 84% のピーク比、
合計 808 µs に対し mma 下限 643 µs）。ただし CUTLASS 2.x のノブ空間は
§10.4 と本節で掃引しきっており、**同じ形状で cuBLAS も同じ約 85% しか出ない**
（p=255 で CUTLASS と wash、本セッションの再測定では cuBLAS が 0.7% 負け）。
85% はライブラリ水準の実効天井であり、ここを越えるのは CUTLASS 3.x / CuTe で
TMA と warp specialization を使う FP64 カーネルを書く作業になる。
**p=255 で `FUSED_TC` との差 6.2 µs（0.6%）を埋める安い手はもう無い。**

**（追記 2026-08-28、`p127_gap_study.md` §13）この「もう無い」は半分だけ正しかった。**
p=127 で z の epilogue を追いかけたところ、mainloop 以外にまだ 3 つ残っていて、
どれも `Nq > 64` の枝なので p=255 にもそのまま効く: `deriv_x` を y GEMM の
2 ソース epilogue に畳んで z の読みを 1 本減らす、lift の積和をアキュムレータの
shared 往復の後ろへ回す、epilogue のアクセスを 16 バイトにする。
**p=255 は 1020.6 → 1004.5 µs/stage（−1.6%）**になった（同日の `FUSED_TC` は
961.0 なので**最速は `FUSED_TC` のまま**、差は 1.045 倍）。
上の結論のうち「CUTLASS のノブ空間は掃引しきっている」「mainloop の 85% は
ライブラリ水準の実効天井」は変わらない。変わったのは、**epilogue 側は
命令数だけでなく「どのカーネルがどの体積テンソルを読むか」という配分にも
余地がある**という点である。詳細と p=127 でのアブレーションは
`p127_gap_study.md` §13。

## 11. Ozaki Scheme I / II（2026-08-28 追記）

同一 DOF（`Ne=1`）で `CUDAFORTRAN_GEMM` を基準に Ozaki 比較経路を計測。
commit `38952e4`、`nstep=100`、`WarmupStep=1`、
`OzakiSliceCount=8` / `OzakiModuliCount=14`、GB200 login ノード。

| 経路 | µs/stage | native GEMM 比 |
|---|---:|---:|
| `CUDAFORTRAN_GEMM` | **3189** | 1.00 |
| `CUDAFORTRAN_GEMM_OZAKI1` | **7020** | **2.2×** |
| `CUDAFORTRAN_GEMM_OZAKI2` | **19.5 ms** | **6.1×** |

[`ozaki2_implementation_report.md`](ozaki2_implementation_report.md) の p=255 単発
測定（OZAKI2 **2.25×**）と整合。最速は `FUSED_TC`（1014 µs/stage）のまま。

> **（訂正 2026-08-29）** 上表の native **3189 µs/stage** は 1 step の device 合計で、
> RK 3 stage で割っていない。1 stage は約 1063 µs。Ozaki 比はそのまま読める。

## 12. 経路横断の再測定（2026-08-29）

[`reports/README.md`](README.md) 用。commit `2dadc41`、login node GPU 1、
3-run 中央値。`conf_perf_p255_tc.conf` / `conf_perf_p255.conf`（`Ne=1`、
`nstep=20`、graph off）の `DqdtKernel_Type` だけを差し替え。µs/stage は
`CUDA device *`。§5 の Main 3.1265 / 1014.2 µs と §10 の 961.0 / 1004.5 は
当時の計測で、表は残す。

| 経路 | Main [ms/step] | µs/stage |
|---|---:|---:|
| `CUDAFORTRAN_FUSED_DFMA` | 6.160 | 1998.0 |
| **`CUDAFORTRAN_FUSED_TC`** | **2.978** | **918.9** |
| `CUDAFORTRAN_GEMM` | 3.441 | 1075.7 |
| `CUDAFORTRAN_GEMM_CUTE` | 3.432 | 1072.9 |
| `CUDAFORTRAN_GEMM_FUSED` | 3.110 | 963.4 |

**最速は `CUDAFORTRAN_FUSED_TC` のまま**（`GEMM_FUSED` に 1.045 倍、§13 と同じ比）。
Main は換算せず実測した。C++ の `FUSED`（DFMA）は 1998 µs で TC 版の 2.17 倍。
この節の経路名は `CUDAFORTRAN_FUSED_DFMA` と読む。
`GEMM` と `GEMM_CUTE` は 0.3% 以内。FLOP/s と DRAM は README のまとめ表。

## 13. CUDA-core 融合の復活（2026-08-29）

`CUDAFORTRAN_FUSED` を Fortran `2dadc41^` の 16×16 タイル x/y/z 3 本として
C++ に戻した。login GPU 1、`conf_perf_p255_tc.conf` の type だけ
`CUDAFORTRAN_FUSED`（`Ne=1`、`nstep=20`、graph off）、3-run 中央値。
作業ツリーは親 `959ad50`。

| 経路 | Main [ms/step] | µs/stage |
|---|---:|---:|
| `CUDAFORTRAN_FUSED`（CC） | 15.755 | 5252.8 |
| `CUDAFORTRAN_FUSED_DFMA`（§12） | 6.160 | 1998.0 |
| `CUDAFORTRAN_FUSED_TC`（§12） | 2.978 | 918.9 |

CC 5252.8 µs は旧 Fortran 〜4970 µs と同水準。fragment 日程の DFMA より遅い。
論文の主比は **TC / FUSED = 918.9 / 5252.8 = 5.72×**。
点変化係数の owned `dqdt` は `FUSED` / `FUSED_DFMA` / `FUSED_TC` が `Ne=1`
（16,777,216 点）で全点ビット一致。`FUSED` 対 `FUSED_TC` は `Ne=2`
（33,554,432 点）でもビット一致。
独立再測（login GPU 1、`conf_perf_p255_tc.conf` の type 差し替え、3-run 中央値）:
FUSED Main 15.75 ms / 5251 µs/stage、DFMA 6.21 ms / 2015 µs/stage、
TC 2.96 ms / 913 µs/stage。表の 5252.8 / 1998.0 / 918.9 は書き換えない
（DFMA の 1% 差はセッション散らばり）。FUSED は DFMA の約 2.6 倍遅いので
経路の取り違えではない。旧 Fortran CC バイナリは残っておらず、〜4970 µs
との約 6% は再確認できない。

## 14. p=255 CC の `launch_bounds`（2026-08-29）

Fortran `2dadc41^` の `tendency_{x,y,z}_p255_kernel` に `launch_bounds` は無い。
添字と shared の写像は C++ と一致する。差は nvcc が
`__launch_bounds__(256,1)` でレジスタを上限まで使ったこと。
login GPU 1、同じ conf、3-run 中央値、点変化係数で TC とビット一致:

| `minBlocks` | Main [ms/step] | µs/stage |
|---:|---:|---:|
| 1（§13） | 15.755 | 5252.8 |
| 4 | 15.755 | 5253.4 |
| **8** | **15.181** | **5058.4** |

`minBlocks=8` が Fortran 〜4970 µs に近い。p=15 は Fortran も bounds 無しだが、
nvcc で外すと 1024 スレッドが起動しないので `(1024,1)` のまま。

## 15. p=255 `CUDAFORTRAN_FUSED` の L1 律速と 2 点/スレッド（2026-08-29）

論文用 CC 対照。最速は `FUSED_TC` のまま。login GPU 1、
`conf_perf_p255_tc.conf` の type だけ `CUDAFORTRAN_FUSED`、`nstep=20`、
graph off、3-run 中央値。ncu は job **66860**（c101、36 s、凍結
`scale-dg_extraction.p255cc_fused`、`output/conf_perf_p255_fused_ncu.conf`）。
作業ツリーは `cd2d77f` 上。§14 の 5058.4 µs は書き換えない。

### 15.1 律速（256 スレッド、1 点/スレッド）

3 本とも **L1/TEX 96.7–98.7%**。DRAM は x/y 2.2–2.7%、z 19%。FMA パイプ
2–3%。占有率 ~99%、32 reg、limit 8 blocks。stall は mio_throttle 15–22、
short_scoreboard 8–22、long 6–12、barrier 6–8。shared ld バンクコンフリクトは
wavefront 比で小さい（x ld 0.72 M / 403 M）。

### 15.2 アブレーション（数値は壊れる。天井だけ）

login GPU 1、同じ conf。正規 INNER=16 は 5058 µs/stage。

| 不正変更 | µs/stage | 対正規 |
|---|---:|---:|
| 内積を DCE で消す（shared も消える） | 380 | 無効な天井 |
| バリア削除 | 4918 | −2.8% |
| global `q*vel` / D1D を 0 | 4468 | −12% |
| INNER=1（shared は生きる） | **2188** | −57% |
| INNER=4 | 2525 | −50% |

K 方向 16 内積が時間の約 57%。バリア賞金は小さい。

### 15.3 採用: 1 スレッド 2 出力（128 スレッド）

x は `dim3(16,8)`、y/z は `dim3(8,16)`。内積で D（x）または D 行（y/z）を
2 出力で共有する。`__launch_bounds__(128, 16)`。

| 形 | Main [ms/step] | µs/stage |
|---|---:|---:|
| 1 点/スレッド（§14） | 15.181 | 5058.4 |
| **2 点/スレッド** | **14.078** | **4684.4** |
| 4 点/スレッド（不採用） | 16.543 | 5519.7 |

2 点は §14 から **−7.4%**。点変化係数で `FUSED` 対 `FUSED_TC` は `Ne=1` /
`Ne=2` とも全点ビット一致。4 点は 64 スレッドで占有が落ち、+18% 遅いので戻した。
論文の主比は当時 **TC / FUSED = 918.9 / 4684.4 = 5.10×**。
2 点版の ncu は job **66862**（凍結 `scale-dg_extraction.p255cc_2w`、c101、35 s）。

### 15.4 2 点/スレッドの律速（job 66862）

L1/TEX は 3 本とも **98.3–98.6%** のまま。占有率 ~99%、32 reg、limit 16 blocks。
ncu 1 起動の時間は x **2.24 ms**、y **3.18 ms**、z **3.18 ms**。x の shared ld は
201 M → 134 M、mio 22 → 6。y/z は D を再利用して体積フラックスは 2 本のまま
（shared ld 201 M、short_scoreboard ~33）。y の shared st は 4-way バンク
（ncu が見積もる上限 ~50%）だが、律速はストアではなく内積の L1 である。

### 15.5 採用: y と z もフラックス側を 2 出力で共有

y の `F_y(i,l,k)` は j に依らない。写像を x と同じ `dim3(16,8)` にし、1 本の
`sQ` 行を 2 つの j に使う。login GPU 1、同じ conf、3-run 中央値
**4201.3 µs/stage**（2 点の 4684.4 から **−10.3%**）、`Ne=1`/`Ne=2` ビット一致。

z の `F_z(line,l)` も k に依らないので同じ写像（2 つの k）。**3697.3 µs/stage**
（y 改修後から **−12.0%**、§14 から **−26.9%**）。Main **11.167 ms/step**。
論文の主比は **TC / FUSED = 918.9 / 3697.3 = 4.02×**。最速は `FUSED_TC` のまま。

| 形 | Main [ms/step] | µs/stage |
|---|---:|---:|
| 1 点（§14） | 15.181 | 5058.4 |
| 2 点、y/z は D 再利用（§15.3） | 14.078 | 4684.4 |
| y もフラックス再利用 | 12.654 | 4201.3 |
| **z もフラックス再利用** | **11.167** | **3697.3** |

この形の ncu は job **66874**（凍結 `scale-dg_extraction.p255cc_qreuse`、c101、37 s）。

### 15.6 フラックス再利用後の律速（job 66874）

3 本の時間が揃った。ncu 1 起動は x **2.24 ms**、y **2.27 ms**、z **2.26 ms**。
いずれも **L1/TEX 96.9–98.3%**、占有率 ~99%、32 reg。DRAM は x/y 3%、z **26%**。
FMA は 3.5–4.3%。stall の主因は x/y が short_scoreboard（30 / 24）、z は
mio_throttle 15 と short 12。shared ld バンクコンフリクトは wavefront 比 ~0.15%。
shared st の 4-way はフラックス再利用で消えた（st コンフリクト比 ~1.4%）。

shared ld 命令は x **134 M**、y/z **201 M**。x の内積は `sQ` 行が連続、y/z は
`sD` 列（stride 16）を読む。

### 15.7 不採用（job 66874 の次）

login GPU 1、同じ conf、3-run 中央値。点変化 `Ne=1`/`Ne=2` は両候補とも
`FUSED_TC` とビット一致。

| 候補 | µs/stage | 対 3697.3 |
|---|---:|---:|
| y の `sD` を転置して内積を行アクセスに | 4493.0 | **+21.5%** |
| タイル末尾バリアを最終反復だけ省略 | 3696.0 | 差なし |

`sD` 転置は内積の stride を消すが、`D1D[j + l*256]` の shared 充填が
stride 256 の global になり、coalesced だった充填より高い。y の `F` は
`i + l*256` なので、充填の連続方向（i）と内積の連続方向（l）は同時に取れない。
x だけが両方できる（`F_x` が `l + j*256`）。y/z の 201 M shared ld は
この転置では埋まらない。

末尾バリアは §15.2 の全削除天井 −2.8% のうち、1/16 反復分で測定誤差以下。
両方戻した。当時の現行は §15.5 の 3697.3 µs。

### 15.8 採用: x の D を shared に載せず `__ldg`（2026-08-29）

login GPU 1、同じ conf、3-run 中央値。作業ツリーは `9a2b887` 上。
x の内積は `D1D[i + l*256]` が i 方向に連続で、`sQ` 行も連続なので、
`sD` を経由すると L1/TEX（§15.6 で 98%）に shared が載るだけだった。
D を内積から `__ldg` し `sQ` だけ shared に残す。

| 形 | Main [ms/step] | µs/stage |
|---|---:|---:|
| z もフラックス再利用（§15.5） | 11.167 | 3697.3 |
| **x の D を global `__ldg`** | **10.821** | **3579.6** |

§15.5 から **−3.2%**。点変化 `Ne=1` / `Ne=2` は `FUSED_TC` と全点ビット一致。
論文の主比は **TC / FUSED = 918.9 / 3579.6 = 3.90×**。最速は `FUSED_TC` のまま。
y も D を global にすると 3896 µs（+8.5%）なので戻した。z は y と同型のため未実装。
`__ldg` は素の `D1D[]` に対し 3580 µs で、login の 0.2% 差（レンジ重なり）だが
読み取り専用キャッシュの意図がコードに残るので採用した。

この形の ncu は job **67347**（c188、31 s、凍結 `scale-dg_extraction.p255cc_xdgldg`、
`output/conf_perf_p255_fused_ncu.conf`）。67261 は `ncu` に nsys 用
`--resolve-symbols=false` を付けて 1 秒で失敗した。

3 本とも **L1/TEX 97.4–99.7%** が屋根。ncu 1 起動は x **2.02 ms**、y **2.28 ms**、
z **2.26 ms**（x が D を global にした分だけ先行）。DRAM は x/y 3.3–3.6%、z **26%**。
占有率 ~99%、32 reg。SASS: x は `LDS.128`×16 + `LDG.E.64.CONSTANT`×16。
y/z は `LDS.64`×48。stall は x が L1TEX scoreboard 38.5 / 53.0 cycle（72.6%）、
y が MIO short scoreboard（shared）24.2 / 47.1 cycle（51.3%）、z が MIO throttle
15.0 / 48.4 cycle（30.9%）。z の ncu 文は「fewer but wider loads」。

### 15.9 分割アブレーション（数値は壊れる。天井だけ）

x-D-global + `__ldg` の 3579.6 µs が分母。login GPU 1、3-run 中央値。
`P255_CC_ABLATE`（本番は 0）。

| 不正変更 | µs/stage | 対 3579.6 |
|---|---:|---:|
| INNER=1 | 1922.4 | −46% |
| D オペランドを 1.0（Q は本物） | 3043.7 | −15.0% |
| Q/vel を 1.0（D は本物） | 3212.1 | −10.3% |
| global 全部 1.0 | 2649.2 | −26.2% |
| エピローグ省略 | 3546.9 | −1.1% |
| 全バリア削除 | 3502.0 | −2.4% |

D と Q の天井はほぼ加算で全部 1.0 の −26% になる。内積の仕事量は CC のままでは
減らせない（INNER=1 は契約外）。エピローグとバリアは測定誤差に近い。

### 15.10 不採用（x-D-global の次。隠蔽は天井を取れない）

login GPU 1、同じ conf、3-run 中央値。点変化 `Ne=1`/`Ne=2` は試した候補で
ビット一致。分母は 3579.6 µs。

| 候補 | µs/stage | 対 3579.6 |
|---|---:|---:|
| x 内積の D を 1 本先読み（レジスタパイプライン） | 3579.6 | 差なし |
| x の次タイル Q を `cp.async` で内積と重ねる | 3811.5 | **+6.5%** |
| y/z の shared を stride 17 に pad | 3580.2 | 差なし |
| D1D 512 KB を stream の persisting L2 window | 3578.5 | 差なし |
| 次タイル Q の `prefetch.global.L2` | 3665.5 | **+2.4%** |
| x の D を `__ldcs`（streaming） | 3675.7 | **+2.7%** |
| 4 点/スレッド（§15.3） | 5519.7 | 占有で敗北（当時の分母は別） |
| x タイル二重バッファ / `__restrict__` / `launch_bounds(128,8)` | （§15.7 前後） | 差なしか遅い |

D を 1.0 にすると −15%、Q を 1.0 にすると −10% だが、それはロード待ちだけではなく
L1/TEX からそのトラフィックを外す。先読み・`cp.async`・prefetch はロードを残したまま
重ねるので、律速資源（L1/TEX）は減らず、命令と shared だけ増えて負けた。
`__ldcs` が遅いのは D がタイル内で再利用され、キャッシュに載せた方が良いから。
pad 17 は §15.6 の st コンフリクト ~1.4% が律速でないことの確認。

範囲外: MMA / TC fragment を `FUSED` に入れる、代表スカラー、INNER を減らす。

### 15.11 律速（job 67347）

**x は L1/TEX 99.7%（発行の 73% が L1TEX scoreboard）。y/z は同じ屋根の下で shared `LDS.64` が MIO を詰まらせる。** DRAM は x の律速ではない。

### 15.12 採用: y の `sD` を j0/j1 の `double2` に（job 67470）

y だけ。`sD[t*16+2*ty]` に `{D(j0,l), D(j1,l)}` を隣接させ内積を `LDS.128` にする
（ncu 67347 が y の MIO short scoreboard と z の「wider loads」を指した）。
SASS は `LDS.64`×48 → `LDS.64`×16（sQ）+ `LDS.128`×16。点変化 `Ne=1`/`Ne=2`
ビット一致。login は 3557.8 µs（−0.61%）でノイズと重なりうるので、c391 で
旧 `p255cc_xdgldg` と新 `p255cc_yd2` を交互 12 回（job **67470**、39 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 旧（x `__ldg`） | 3585.5 | 3584.8–3586.2 |
| **y `double2`** | **3563.7** | 3562.8–3564.3 |

レンジは重ならない。**−0.61%**。機構は shared D のロードが 2 本の `LDS.64` から
1 本の `LDS.128` になったこと。

### 15.13 採用: z も同じ `double2`（job 67529）

login 3-run **3546.3 µs** / Main 10.723 ms。SASS は y と同じ `LDS.64`×16 +
`LDS.128`×16。点変化 `Ne=1`/`Ne=2` ビット一致。c384 で `p255cc_yd2` 対
`p255cc_yzd2` を交互 12 回（job **67529**、53 s）。旧の 9 回目 3565.6 は外れ値でも
レンジは重ならない。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| y `double2` のみ | 3560.9 | 3560.0–3565.6 |
| **y+z `double2`** | **3549.4** | 3548.2–3550.1 |

**−0.32%**。x `__ldg` の 3585.5（job 67470）から **−1.01%**。機構は y と同じ。
次は手順 3: 凍結 `p255cc_yzd2` の ncu（job **67606**）。

### 15.14 律速（job 67606、c390、29 s）

ncu 時間は採否に使わない。機構: x は変わらず L1/TEX 99.6%、L1TEX scoreboard
38.4 / 53.0 cycle（72.4%）、2.02 ms。y は 2.28 → 2.24 ms、命令 503 M → 455 M、
MIO short 24.2 / 47.1（51%）→ 20.0 / 51.3 cycle（39%）。z は MIO throttle が消え
MIO short 17.4 / 56.5 cycle（31%）、命令 486 M → 413 M。y/z に残る `LDS.64` は sQ。
屋根はなお L1/TEX 98.8–99.6%。

### 15.15 不採用: y の sQ も `{F(t), F(t+8)}` の `double2`（job 67772）

内積を t と t+8 で畳む（加算順が変わる）。対 `FUSED_TC` の max abs は 2.6e-13
（丸め）。SASS は `LDS.128`×24 + `STS.128`、`DFMA` 36→20。login は 3549.9 µs
（現行 login 3546.3 から +0.10%）。c188 で `p255cc_yzd2` 対 `p255cc_yqd2` を
交互 12 回（job **67772**）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| y+z `sD` `double2`（現行） | 3547.1 | 3546.6–3547.8 |
| y `sQ` も `double2`（畳み） | 3549.7 | 3548.8–3550.9 |

レンジは重ならない。**+0.07%**。戻した。屋根は L1/TEX 99% のままなので、
残る sQ の `LDS.64` を広くしても MIO は律速ではなく、`STS.128` と内積の畳みの
方が高い。同じ詰めを z にやるのは同一仮説なのでやらない。畳まず 16 回のまま
`double2` だけ載せる版は §15.72 で +13.9%。

### 15.16 不採用: x が 2 つの `tile_j` で D を shared 再利用

D は `D(i,l)` で j に依らない。現行は `tile_j` ごとに別ブロックが同じ D を
`__ldg` する。1 ブロックが隣接 2 `tile_j`（4 出力）を持ち、`sD` を 1 タイル
載せて 2 回使う。グリッドは 65536 → 32768。y/z は変えない。
天井は D オペランド 1.0 の −15%（§15.9）のうち x が担う分。sD を 1 回使い
に戻すだけなら §15.8 で負けているので、再利用が dual-issue の条件である。

login 3-run は device **7144 µs/stage**（現行 3546 の約 2.0 倍）。点変化 `Ne=1` /
`Ne=2` は現行と全点一致（max abs 0）。4 本のアキュムレータで占有が落ちた。戻した。
占有 GPU の A/B は不要（差が 1% を遥かに超える）。

### 15.17 不採用: y エピローグの `dqdt` を `cp.async` 先読み（job 67984）

p=63 §19.4 / p=31 で効いた。天井はエピローグ全体の −1.1%（§15.9）。y だけ。
sm_100 の `cp.async` は 16 B なので、偶数 `tx` が隣接 i の 2 点を 1 本で運び、
wait のあと追加 `__syncthreads` した。点変化 `Ne=1`/`Ne=2` は現行と全点一致。
c386 で `p255cc_yzd2` 対 `p255cc_ycp` を交互 12 回（job **67984**、51 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| y+z `sD` `double2`（現行） | 3551.8 | 3550.5–3553.9 |
| y `dqdt` `cp.async` | 3559.0 | 3557.5–3560.8 |

レンジは重ならない。**+0.20%**。戻した。屋根は L1/TEX なのでエピローグの
RMW を shared に移しても L1 は減らず、追加バリアと `cp.async` が乗る。
z に同じ（追加バリア付き）は同一仮説なのでやらない。追加バリア無しは §15.18。

### 15.18 不採用: 同じ `cp.async` を追加バリア無しで（job 68001）

奇数 `tx` も 16 B 整列アドレスへ自分で `cp.async` し、`wait_group 0` のあと
自分のスロットだけ読む。peer 用 `__syncthreads` は付けない。
点変化 `Ne=1`/`Ne=2` は全点一致。c188 で `p255cc_yzd2` 対 `p255cc_ycp2` を
交互 12 回（job **68001**、48 s。`squeue` で完了を確認）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3547.4 | 3546.9–3548.0 |
| 追加バリア無し `cp.async` | 3559.2 | 3558.4–3559.6 |

レンジは重ならない。**+0.34%**。戻した。追加バリアが原因ではなく、
`dqdt` を L1 に載せる先読みそのものが屋根を奪う。z の同手は同一仮説なのでやらない。

### 15.19 不採用: z の Q/vel を `__ldcs`（job 68012）

ncu 67606 で z だけ DRAM 26%。Q は `l` 方向に stride 65536 で一度しか使わない。
x の D に `__ldcs` したときは再利用があるので負けた（§15.10、+2.7%）。z の Q は
ストリーミング側なので別仮説。点変化 `Ne=1`/`Ne=2` は全点一致。c188 で
`p255cc_yzd2` 対 `p255cc_zldcs` を交互 12 回（job **68012**、47 s。`squeue` で確認）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3546.0 | 3544.9–3546.5 |
| z `__ldcs` | 3546.1 | 3545.3–3546.6 |

レンジは重なる。**差が無い**。戻した。z の DRAM 26% は ncu のクロック固定で
大きく見えるが、実時間ではこのヒントは効かない。

### 15.20 不採用: タイルバリアを先頭へ（job 68025）

全削除の天井は −2.4%（§15.9）。§15.7 の「末尾だけ最終反復で省略」は login で
差なし。p=63 は**位置**を末尾から先頭へ移して −7.7%。x/y/z とも
`if (ltile > 0) sync; load; sync; inner`（末尾 sync なし）。点変化 `Ne=1`/`Ne=2`
は全点一致。login は 3560–3562 µs。c188 で交互 12 回（job **68025**、51 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3547.4 | 3547.0–3548.5 |
| バリア先頭 | 3562.3 | 3561.5–3562.7 |

レンジは重ならない。**+0.42%**。戻した。全削除天井 −2.4% のうち位置の組み替えは
L1 律速では負ける。p=63 は同期点が律速だった。

### 15.21 不採用: 引数に `__restrict__` だけ（job 68028）

§15.10 では二重バッファ等と束ねて「差なし」。今回はポインタ修飾だけ。
点変化 `Ne=1`/`Ne=2` は全点一致。c188 で交互 12 回（job **68028**、44 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3547.0 | 3546.2–3548.9 |
| `__restrict__` | 3545.4 | 3544.9–3546.2 |

中央値は −0.05% だがレンジが重なるので **差が無い**。戻した。p=15/p=63 と違い
p=255 CC は 32 レジスタ固定で、restrict が開けるスケジューリング余地が無い。

### 15.22 不採用: y の Q/vel を `__ldg`（job 68034）

x の D は `__ldg` で採用済み。y の Q はタイル内で shared 再利用する前に一度読む。
点変化 `Ne=1`/`Ne=2` は全点一致。c182 で交互 12 回（job **68034**、52 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3549.6 | 3548.7–3550.7 |
| y Q `__ldg` | 3552.1 | 3551.2–3552.7 |

レンジは重ならない。**+0.07%**。戻した。定数キャッシュ向けの `__ldg` は D には効くが
点変化の Q には効かない。z の Q に同じことはやらない。

### 15.23 不採用: `__launch_bounds__(128, 12)`（job 68061）

`(128, 16)` が現行、`(128, 8)` は遅かった（§15.10）。12 は未測。
点変化 `Ne=1`/`Ne=2` は全点一致。c395 で交互 12 回（job **68061**、48 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| `(128, 16)` | 3546.9 | 3546.0–3547.5 |
| `(128, 12)` | 3548.8 | 3548.1–3549.2 |

レンジは重ならない。**+0.06%**。戻した。占有 16 がこの屋根では下限ではない。

### 15.24 不採用: ltile ループに `#pragma unroll`（job 68125）

点変化 `Ne=1`/`Ne=2` は全点一致。login は 3632 µs。c396 で交互 12 回（job
**68125**、48 s）。現行 3551 前後に対し新 3639 前後、**約 +2.5%**。戻した。
16 回展開は命令キャッシュとレジスタを食う。

### 15.25 不採用: 末尾バリアを最終タイルだけ省略（job 68227）

§15.7 は 3697 µs の login 3-run で差なし。全削除天井 −2.4% の 1/16。
x/y/z の内積後 sync を `ltile < 15` に限る。ロード後 sync は残す。
点変化 `Ne=1`/`Ne=2` は全点一致（max abs 0）。login は 3545–3546 µs。
c182 で交互 12 回（job **68227**、48 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 yzd2 | 3549.9 | 3548.5–3551.7 |
| 最終タイルだけ省略 | 3549.5 | 3548.4–3550.8 |

レンジは重なる。中央値差 **−0.01%**。差なし。戻した。

### 15.26 不採用: x の D を `__ldcg`（job 68233）

`__ldg` は採用、`__ldcs` は +2.7%（§15.10）。`__ldcg`（L2 / cache-global）は未測だった。
点変化 `Ne=1`/`Ne=2` は全点一致。login は 3551 µs。c180 で交互 12 回（job
**68233**、42 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| `__ldg` | 3545.7 | 3544.9–3546.2 |
| `__ldcg` | 3551.2 | 3550.6–3552.6 |

レンジは重ならない。**+0.16%**。戻した。読み取り専用キャッシュ向けの `__ldg` がこの D には合う。

### 15.27 不採用: 内積ループを `#pragma unroll 4`（job 68244）

現行は無指定＝16 展開。点変化 `Ne=1`/`Ne=2` は全点一致。login は 3684 µs。
c182 で交互 12 回（job **68244**、50 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| `#pragma unroll` | 3548.3 | 3547.4–3550.5 |
| `#pragma unroll 4` | 3684.7 | 3683.1–3686.2 |

レンジは重ならない。**+3.85%**。戻した。§15.24 の外ループ全展開と同様、展開を削ると内積が L1 屋根の下で伸びる。

### 15.28 不採用: x の sQ を `{F(j0),F(j1)}` の `double2`（job 68311）

ncu 67606 の残 `LDS.64` は sQ。y/z の sD と同じく 2 点を隣接 `double2` にした。
点変化 `Ne=1`/`Ne=2` は全点一致。login は 3931 µs。c183 で交互 12 回（job
**68311**、44 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3546.7 | 3545.9–3547.6 |
| x sQ `double2` | 3931.3 | 3930.5–3932.0 |

レンジは重ならない。**+10.8%**。戻した。行が `t` の packed レイアウトは
連続 `tx` が 16 double ストライドで同じバンクに STS する。

### 15.29 yzd2 カーネルでの分割アブレーション再測（job 68308）

分母は同一ジョブの yzd2 3539.6 µs（c182、3-run 中央値）。数値は壊れる。

| 不正変更 | µs/stage | 対 3539.6 | 対 §15.9（3579.6） |
|---|---:|---:|---:|
| INNER=1 | 1807.7 | −48.9% | −46% |
| D オペランドを 1.0 | 3009.9 | −15.0% | −15.0% |
| Q/vel を 1.0 | 3199.2 | −9.6% | −10.3% |
| global 全部 1.0 | 2636.3 | −25.5% | −26.2% |
| エピローグ省略 | 3496.8 | −1.2% | −1.1% |
| 全バリア削除 | 3472.4 | −1.9% | −2.4% |

yzd2 採用後も天井の形は同じ。INNER / D / Q は測定誤差より遥かに大きいが、
INNER 削減は契約外。D と Q の −15%/−10% は L1 から仕事を消す天井で、隠蔽では
取れない（§15.10）。エピローグとバリアは 2% 未満。

### 15.30 不採用: x sQ `double2` に列 XOR スウィズル（job 68332）

§15.28 の 16-way STS を `col = 2*(ty^(tx&7))` で散らす。点変化は全点一致。
login は 3641 µs。c182 で交互 12 回（job **68332**、50 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3548.2 | 3546.7–3549.5 |
| packed + XOR | 3643.8 | 3642.2–3647.8 |

レンジは重ならない。**+2.7%**。バンクは直っても packed 行レイアウト自体が負け。戻した。

### 15.31 不採用: x の D を `__ldca`（job 68395）

`__ldg` 採用、`__ldcs` +2.7%、`__ldcg` +0.16%。全レベルキャッシュは未測だった。
点変化は全点一致。c397 で交互 12 回（job **68395**、43 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| `__ldg` | 3543.6 | 3541.1–3545.9 |
| `__ldca` | 3550.6 | 3547.6–3553.4 |

レンジは重ならない。**+0.20%**。戻した。

### 15.32 不採用: x 内積を既存 sQ 行の連続 `t` で `double2`（job 68419）

レイアウトは据え置き、`t+=2` で `LDS.128`。加算順が変わる。点変化 `Ne=1`/`Ne=2`
max abs 1.2e-13 / 1.6e-13。login は 3546 µs。c147 で交互 12 回（job **68419**、
46 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3545.6 | 3544.6–3551.9 |
| `t+=2` `double2` | 3546.2 | 3545.2–3547.1 |

レンジは重なる。**差が無い**。戻した。x の残 `LDS.64` はブロードキャストが多く、
屋根は D の L1 なので sQ を広くしても動かない。

### 15.33 不採用: x の Q/vel 充填を `__ldcs`（job 68513）

一度きりストリーミング。y の `__ldg` は +0.07%、z の Q `__ldcs` は差なし。
x の Q は未測。点変化は全点一致。c180 で交互 12 回（job **68513**、42 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3548.1 | 3547.6–3548.7 |
| x Q `__ldcs` | 3550.0 | 3549.3–3551.2 |

レンジは重ならない。**+0.05%**。戻した。x の Q も L1 屋根の下ではストリーミングが負ける。

### 15.34 不採用: x の D を `__ldlu`（job 68514）

last-use。D はタイル内で 16 回再利用されるので不向きなはず。点変化は全点一致。
login は 3646 µs。c180 で交互 12 回（job **68514**、42 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| `__ldg` | 3548.4 | 3547.7–3548.9 |
| `__ldlu` | 3648.7 | 3635.0–3654.1 |

レンジは重ならない。**+2.8%**。戻した。キャッシュヒントは `__ldg` だけが勝つ。

### 15.35 不採用: `__launch_bounds__(128, 20)`（job 68515）

128×20=2560 は SM の 2048 スレッド上限を超える。ptxas は
`.minnctapersm will be ignored`。点変化は全点一致。c180 で交互 12 回（job
**68515**、41 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| `(128, 16)` | 3548.4 | 3547.8–3549.1 |
| `(128, 20)`（無視） | 3551.8 | 3551.1–3552.3 |

レンジは重ならない。**+0.10%**。戻した。16 がハードウェア上限で、ヒントを外すと
スケジューラが悪化する。

### 15.36 不採用: x エピローグの Lift1D を `__ldg`（job 68517）

天井はエピローグ −1.2%。点変化は全点一致。c182 で交互 12 回（job **68517**、
37 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3539.7 | 3538.1–3540.8 |
| Lift `__ldg` | 3539.6 | 3538.8–3541.1 |

レンジは重なる。**差が無い**。戻した。Lift は 1.5 KB で既に L1 に載っている。

### 15.37 不採用: y の sD 充填を `__ldg`（job 68522）

x 内積の `__ldg` は採用済み。y は D を shared に載せる。点変化は全点一致。
c183 で交互 12 回（job **68522**、39 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3549.1 | 3548.4–3549.8 |
| y sD `__ldg` | 3551.7 | 3550.8–3553.0 |

レンジは重ならない。**+0.07%**。戻した。z の同手は同一仮説なのでやらない。

### 15.38 不採用: 内積 `#pragma unroll 8`

§15.27 の 4 と現行 16 の間。点変化は全点一致。login 3-run **4375 µs**（現行
3546 の **+23%**）。占有 A/B は不要。戻した。部分展開は 4 より悪い。

### 15.39 採用: x エピローグの Escale を `__ldg`（job 68524）

天井はエピローグ −1.2%。点変化 `Ne=1`/`Ne=2` は全点一致。login は 3543–3545 µs。
c180 で交互 12 回（job **68524**、40 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 yzd2 | 3545.4 | 3544.9–3546.3 |
| **x Escale `__ldg`** | **3543.6** | 3542.4–3544.2 |

レンジは重ならない。**−0.05%**。点変化の Q への `__ldg`（§15.22）とは逆で、
エピローグの 1 回読みを定数キャッシュに逃がすと L1 屋根の内積側がわずかに空く。
機構の ncu は同一ジョブで yzd2 と並べて採る。

ncu job **68526**（c180、77 s、`--set full`、両 exe 同一ジョブ）: x の Duration は
ncu 時間なので採否に使わない。L1/TEX は 99.66→99.65% のまま屋根。x の
L1/TEX hit 55.95→55.41%。定数キャッシュへ逃がした分、L1 のヒット率計算の
分母が内積の D/sQ に寄る。屋根は動かない。−0.05% はエピローグの発行が
L1 待ちの隙間に入った分。

### 15.40 不採用: y エピローグの Escale を `__ldg`（job 68527）

x と同じ手。点変化は全点一致。c180 で交互 12 回（job **68527**、39 s）。
分母は §15.39 の xesc。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| x Escale `__ldg` | 3543.3 | 3542.7–3543.9 |
| y も Escale `__ldg` | 3548.6 | 3547.9–3549.5 |

レンジは重ならない。**+0.15%**。戻した。y は RMW の `dqdt` が L1 を既に
使っており、追加の定数キャッシュ指示が負ける。z の同手は同一仮説なのでやらない。

### 15.41 不採用: x エピローグの `flux_bnd` を `__ldg`（job 68532）

点変化は全点一致。c178 で交互 12 回（job **68532**、39 s）。分母は xesc。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| x Escale `__ldg` | 3544.9 | 3544.0–3546.2 |
| `flux_bnd` `__ldg` | 3547.8 | 3547.2–3548.9 |

レンジは重ならない。**+0.08%**。戻した。面フラックスは点変化で定数キャッシュ向きでない。

### 15.42 不採用: x の `dqdt` を `__stwb`（job 68549）

y の RMW が L1 に残るように write-back。点変化は全点一致。c183 で交互 12 回
（job **68549**、39 s）。分母は xesc。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3547.2 | 3546.4–3547.8 |
| `__stwb` | 3547.1 | 3546.5–3547.8 |

レンジは重なる。**差が無い**。戻した。x の素のストアは既に L1 に載っている。

### 15.43 不採用: x の `dqdt` を `__stcs`（job 68550）

ストリーミング（逆仮説）。点変化は全点一致。c183 で交互 12 回（job **68550**、
39 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3547.1 | 3546.3–3548.1 |
| `__stcs` | 3547.2 | 3546.5–3547.8 |

レンジは重なる。**差が無い**。戻した。ストアヒント 2 種ともエピローグ天井を動かさない。

### 15.44 不採用: y の `dqdt` RMW 読みを `__ldcs`（job 68552）

`cp.async` 先読み（§15.17–15.18）とは別。点変化は全点一致。c183 で交互 12 回
（job **68552**、39 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3547.0 | 3546.5–3547.6 |
| y `dqdt` `__ldcs` | 3546.6 | 3546.0–3547.7 |

レンジは重なる。**差が無い**。戻した。z の同手は同一仮説なのでやらない。

### 15.45 不採用: x の `dqdt` を `__stcg`（job 68553）

残りのストアヒント。点変化は全点一致。c183 で交互 12 回（job **68553**、39 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行 | 3547.1 | 3546.8–3547.9 |
| `__stcg` | 3547.1 | 3546.0–3547.8 |

レンジは重なる。**差が無い**。戻した。`__stwb` / `__stcs` / `__stcg` はすべて差なし。
`__stwt` は同一族なのでやらない。

### 15.46 不採用: x だけ `dim3(8,16)`（job 68591）

現行 `dim3(16,8)` はワープが 16 本の i を持つ。`dim3(8,16)` にすると
`tid = tx + 8*ty`、`ix = tid & 15`、`jy = tid >> 4` でワープが i と j を混ぜる。
点変化は全点一致。login は 3546–3548 µs。c183 で交互 12 回（job **68591**、40 s）。
分母は §15.39 の xesc。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| `dim3(16,8)` | 3547.1 | 3546.5–3547.7 |
| `dim3(8,16)` | 3550.3 | 3549.7–3551.2 |

レンジは重ならない。**+0.09%**。戻した。y/z の同形は x が負けたのでやらない。

### 15.47 不採用: `cudaSharedmemCarveoutMaxL1`

L1/TEX が屋根で shared は x 2 KB・y/z 4 KB。p=7 TC では
`CarveoutMaxShared` が L1 を削って +31% だった。逆側の `MaxL1` を 3 本に
`cudaFuncSetAttribute`（初回起動のみ）。点変化は全点一致。login GPU 1 で xesc は 3544.1 µs。c183 で交互 12 回
（job **68637**、53 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| 現行 | 3547.0 | 3546.5–3547.6 |
| `MaxL1` | 19164.7 | 19125.8–19256.7 |

レンジは重ならない。**+441%**。戻した。既定 carveout がすでに L1 寄りで、
`MaxL1` はブロック数か L1/shared 分割を壊す。

### 15.48 不採用: x 内積の D を 1 段ソフトウェアパイプライン（job 68652）

`__ldg` の D を `t` と `t+1` で重ねる。交通量は同じ。点変化は全点一致。
login は 3544.0–3544.5 µs。c398 で交互 12 回（job **68652**、39 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| 現行 | 3547.5 | 3547.0–3548.1 |
| D パイプライン | 3547.5 | 3546.8–3548.1 |

レンジは重なる。**差が無い**。戻した。L1 屋根では次の D を先に発行しても
待ちは隠れない（§15.10 の隠蔽天井と一致）。

### 15.49 採用: x だけ l タイル 16→32（job 68657）

内積は 32 連続の `l`、sQ は 16 j × 32 l（4 KB）。バリアは 16→8 回。D の
L1 行が 2 倍長く載る。y/z は変えない。INNER 本数は 256 のまま。
点変化 `Ne=1`/`Ne=2` は全点一致。login 3-run 中央値 **10.686 ms / 3535.1 µs**。
c183 で交互 12 回（job **68657**、38 s）。分母は §15.39 の xesc。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| xesc（l タイル 16） | 3547.1 | 3546.5–3548.0 |
| **x l タイル 32** | **3538.3** | 3537.7–3538.9 |

レンジは重ならない。**−0.25%**。採用。

ncu job **68658**（c183、79 s、`--set full`、xesc と xk32 を同一ジョブ）:
x の Duration（ncu 時間、採否に使わない）は 2027 → 2009 µs。L1/TEX は
99.67→99.72% のまま屋根。x の L1 hit 55.37→54.63%。y/z は Duration・屋根とも
不変。占有率 99%、32 reg のまま。効いたのは x のバリア 16→8 と、32 連続の
`D(i,l)` が L1 に長く載ること。屋根は動かないので −0.25% が上限に近い。

### 15.50 不採用: y だけ l タイル 16→32（job 68909）

x と同手。sD/sQ を 32 行。点変化は全点一致。login は 3544–3545 µs。
c111 で交互 12 回（job **68909**、39 s）。分母は §15.49 の xk32。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| x l タイル 32 | 3534.3 | 3531.8–3535.9 |
| y も l タイル 32 | 3543.4 | 3542.3–3544.4 |

レンジは重ならない。**+0.26%**。戻した。y は D が shared で、行を 2 倍にすると
充填と内積の shared が増える。z の同手は同一仮説なのでやらない。

### 15.51 不採用: x だけ l タイル 32→64（job 68912）

sQ 8 KB、内積 64、バリア 4 回。点変化は全点一致。login は 3537–3538 µs。
c398 で交互 12 回（job **68912**、48 s）。分母は §15.49。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| l タイル 32 | 3538.8 | 3537.6–3539.8 |
| l タイル 64 | 3541.5 | 3540.0–3542.7 |

レンジは重ならない。**+0.08%**。戻した。32 が谷で、64 は shared 8 KB と
展開 64 が L1 屋根の上に乗る。

### 15.52 不採用: x と y を 1 カーネルに融合（job 68947）

同一 `(tile_i, tile_j, k)` で x のあと y を回し、y の RMW を x のストアに畳む。
`q` の再利用にはならない: x は `q(:,j,k)`、y は `q(i,:,k)` で、交差は 16×16 点
（各カーネル 4096 点の 6%）だけ。天井はエピローグ −1.2% のうち y の RMW 分。
点変化は全点一致。login は 3535.9–3536.4 µs。c111 で交互 12 回（job **68947**、
38 s）。分母は §15.49 の xk32。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| 分離 x+y | 3532.8 | 3532.2–3535.9 |
| xy 融合 | 3533.9 | 3533.1–3534.5 |

レンジは重なる。**差が無い**。戻した。L1 内積が支配的で、起動 1 本と RMW 1 回
は測定誤差以下。z はグリッドが `line×k` なので同型の融合は別写像になる。

### 15.53 不採用: z エピローグの Escale を `__ldg`（job 68948）

x では −0.05%、y では +0.15%。z は DRAM 26% で屋根が違う。点変化は全点一致。
c111 で交互 12 回（job **68948**、38 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| 現行 | 3533.0 | 3531.6–3535.2 |
| z Escale `__ldg` | 3532.4 | 3531.2–3533.5 |

レンジは重なる。**差が無い**。戻した。エピローグは z の DRAM 待ちの陰に入る。

### 15.54 不採用: z の Q/vel 充填を `__ldg`（job 68949）

z の Q `__ldcs` は差なし（§15.19）。定数キャッシュは未測。DRAM 26% の z だけ。
点変化は全点一致。c111 で交互 12 回（job **68949**、38 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| 現行 | 3533.1 | 3531.8–3534.5 |
| z Q/vel `__ldg` | 3533.5 | 3532.7–3534.6 |

レンジは重なる。**差が無い**。戻した。z の Q は再利用が無く、`__ldg` は効かない。

### 15.55 不採用: z だけ `__launch_bounds__(128, 8)`（job 68969）

p=63 の y はブロック数を落として DRAM 待ちを隠した。z は DRAM 26%。x/y は 16 のまま。
点変化は全点一致。login は 3623–3634 µs。c183 で交互 12 回（job **68969**、52 s）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| `(128, 16)` | 3538.6 | 3538.0–3540.0 |
| z だけ `(128, 8)` | 3628.9 | 3626.0–3632.5 |

レンジは重ならない。**+2.6%**。戻した。z も L1/TEX 98.8% が屋根で、占有を半減すると
DRAM 待ちよりスケジュールが悪化する。

### 15.56 不採用: x の Q/vel 充填を `__ldg`（job 68970）

`__ldcs` はタイル 16 で +0.05%（§15.33）。タイル 32 の `__ldg` は未測だった。
点変化は全点一致。c183 で交互 12 回（job **68970**）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| 現行 | 3538.8 | 3538.0–3539.6 |
| x Q/vel `__ldg` | 3538.6 | 3538.1–3539.3 |

レンジは重なる。**差が無い**。戻した。x の Q は shared に載ったあと内積され、
充填ヒントは屋根を動かさない。

### 15.57 不採用: z が 2 `line` タイルで D を逐次再利用（job 68972）

z の `D(k,l)` は `line` に依らない。グリッド半減、4 アキュムレータ、`sQ` を
2 本の line で逐次上書き（バリアが 2→3 本/タイル）。点変化は全点一致。
c183 で交互 12 回（job **68972**）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| 現行 | 3538.7 | 3537.9–3539.4 |
| 逐次 2 line | 3538.5 | 3537.8–3539.5 |

レンジは重なる。**差が無い**。D を半減した分が追加バリアと内積 2 回に食われる。

### 15.58 採用: z の 2 `line` を dual `sQ` で同時内積（job 68973）

§15.57 と同じグリッド半減・D 1 回充填だが、`sQ0`/`sQ1` を同時に載せて
バリア本数は現行と同じ 2 本/タイル。内積は 1 本の D に 2 本の F を掛ける。
点変化は全点一致。login 3-run **9.927 ms / 3276.9 µs**。c183 で交互 12 回
（job **68973**）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| xk32 | 3538.5 | 3538.0–3539.4 |
| **z dual sQ 2 line** | **3279.3** | 3276.9–3282.4 |

レンジは重ならない。**−7.3%**。採用。

ncu job **68975**（c183、`--set full`、xk32 と zd2q 同一ジョブ）: z の Duration
（ncu 時間、採否に使わない）は 2242 → 1740 µs。L1/TEX は 98.8→98.4% のまま屋根。
z の DRAM は 26.5→34.1%（ブロック半減で Q のユニーク率が上がる）。L1 hit は
24.7→9.8%。x/y は不変。占有率 97%、32 reg。効いたのは D 充填の 4096 重の除去で、
屋根は L1 のまま z の仕事が減った。

### 15.59 採用: y も隣接 2 `tile_i` で dual `sQ`（job 68976）

y の `D(j,l)` は i に依らない。z と同じ dual `sQ`。点変化は全点一致。
login 3-run **9.767 ms / 3223.6 µs**。c183 で交互 12 回（job **68976**）。
分母は §15.58 の zd2q。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| z dual sQ | 3279.0 | 3277.0–3284.8 |
| **+ y dual sQ** | **3227.6** | 3226.3–3232.5 |

レンジは重ならない。**−1.6%**。採用。y の ncu Duration は z より元から短く、
冗長 D の絶対量が z より小さい。

### 15.60 採用: x も隣接 2 `tile_j` で dual `sQ`（job 68996）

x の `D(i,l)` は j に依らない。§15.16 は D を shared に載せて 4 出力にし占有が
落ちた。今回は `__ldg` の D を保ったまま `sQ0`/`sQ1`（各 `32*16`）と 4 アキュムレータ。
グリッドは y/z と同じ 32768/要素。点変化 `Ne=1`/`Ne=2` は全点ビット一致。
login 3-run **9.496 ms / 3130.8 µs**。c183 で交互 12 回（job **68996**）。
分母は §15.59 の yd2q。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| y dual sQ | 3227.6 | 3226.6–3233.4 |
| **+ x dual sQ** | **3131.4** | 3128.2–3144.4 |

レンジは重ならない。**−3.0%**。採用。

ncu job **69046**（c183、`--set full`、yd2q と xd2q 同一ジョブ）: x の Duration
（ncu 時間、採否に使わない）は 2011 → 1694 µs。L1/TEX は 99.7→98.4% のまま屋根。
x の DRAM は 3.27→3.88%。L1 hit は 54.7→40.7%。占有率 98%、32 reg、static
shared 4→8 KB。y/z は不変。効いたのは同一 `D(i,l)` の `__ldg` を 16 `tile_j`
から 8 pair に半減したこと。

### 15.61 不採用: z の 4 line を 4 枚 `sQ` で同時内積（job 69048）

2-way のあとに残る同一 `tile_k` の D 充填冗長をさらに半減する。`sQ0`–`sQ3`、
8 アキュムレータ、グリッド 16384/要素。点変化は全点ビット一致。
login は 3821–3832 µs。c183 で交互 12 回（job **69048**）。分母は §15.60 の xd2q。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| x dual sQ | 3131.8 | 3129.4–3136.5 |
| z 4-line 4 sQ | 3816.7 | 3812.1–3826.6 |

レンジは重ならない。**+21.9%**。戻した。内積は L1 屋根のまま Q パネルが倍になり、
D 充填を半減した分を共有メモリの内積負荷が上回る。当時は y の 4 `tile_i` と x の
4 `tile_j` を同じ形として測らなかった。実測は §15.69 / §15.70。

### 15.62 不採用: x 内積の D `__ldg` をワープ内で 1 回に（job 69054）

`dim3(16,8)` のワープは `(ty even, tx)` と `(ty odd, tx)`。偶 ty だけ `__ldg` し
`__shfl_sync(..., tx)` で配る。点変化は全点ビット一致。login 4034–4037 µs。
c183 で交互 12 回（job **69054**）。分母は §15.60 の xd2q。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| x dual sQ | 3132.1 | 3125.5–3141.7 |
| D shuffle | 4038.6 | 4037.3–4040.2 |

レンジは重ならない。**+28.9%**。戻した。L1 屋根の内積に shfl を足すと D 削減より
同期の方が高い。xor-exchange は偶 ty の D を潰すので数値も壊れる（max abs ~6.6）。

### 15.63 採用: x の `sQ` 行ストライド 32→33（job 69063）

ncu job **69046** は x の shared **store** が平均 2.4-way バンクコンフリクト
（wavefront の ~17%）。行長 33 で `(ty*33+t) mod 32` が ty に依る。点変化は全点
ビット一致。login 3-run **9.392 ms / 3095.6 µs**。c183 で交互 12 回（job **69063**）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| x dual sQ stride 32 | 3136.5 | 3128.8–3140.6 |
| **stride 33** | **3098.7** | 3096.2–3102.1 |

レンジは重ならない。**−1.2%**。採用。ncu job **69087**（c387、xd2q と xpad 同一ジョブ）:
shared store コンフリクト 3.55 M → 1.94 M（2.4-way → 2.2-way）。ncu の x Duration は
1689 → 1746 µs と**増える**（クロック固定のいつものバイアス）。壁時計は勝つ。
残コンフリクトは連続 ty がバンク 1–15 で重なること。

### 15.64 不採用: x `sQ` ストライド 48（job 69088）

stride%32=16 で連続 ty のバンクを 0–15 と 16–31 に分ける。shared は 8→12 KB。
点変化は全点ビット一致。login 3138–3153 µs。c387 で交互 12 回（job **69088**）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| stride 33 | 3097.3 | 3095.8–3103.5 |
| stride 48 | 3146.8 | 3131.0–3152.9 |

レンジは重ならない。**+1.6%**。戻した。残 2.2-way を消すよりパネル肥大の方が高い。

### 15.65 不採用: x `sQ` 列を `tx ^ ((ty&1)*16)`（job 69537）

ストライドは 33 のまま、ワープ内の偶/奇 ty をバンク 0–15 と 16–31 に分ける。
点変化は全点ビット一致。login 3355–3357 µs。c384 で交互 12 回（job **69537**）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| stride 33 | 3094.4 | 3093.6–3097.6 |
| 列 XOR | 3357.6 | 3355.8–3375.4 |

レンジは重ならない。**+8.5%**。戻した。内積のアドレス計算が L1 屋根に勝つ。

### 15.66 現行（stride 33 dual sQ）の分割アブレーション（job 69535）

分母は同一ジョブの xpad **3094.1 µs**（c384、3-run 中央値）。数値は壊れる。

| 不正変更 | µs/stage | 対 3094.1 | 対 §15.29（3539.6） |
|---|---:|---:|---:|
| INNER=1 | 1681.3 | **−45.7%** | −48.9% |
| D オペランドを 1.0 | 2623.6 | **−15.2%** | −15.0% |
| Q/vel を 1.0 | 2453.9 | **−20.7%** | −9.6% |
| global 全部 1.0 | 2106.7 | **−31.9%** | −25.5% |
| エピローグ省略 | 3557.2 | +15.0% | −1.2% |
| 全バリア削除 | 3078.3 | −0.5% | −1.9% |

dual sQ のあと **Q 天井が −10% から −21% に上がった**。x は 16 個の `tile_i`
ブロックが同じ `(j,k)` の Q を読み直す。INNER は契約外。D は −15% のまま
（内積の `__ldg` が残る）。バリアは誤差。エピローグ省略が遅く出るのは
コード生成の副作用で、賞金ではない。

### 15.67 不採用: x が 2 `tile_i` で同一 `sQ` を使う（job 69542）

Q は i に依らない。パネルは増やさず 2 本の i に D を掛け、8 アキュムレータ。
グリッド 16384/要素。点変化は全点ビット一致。login 4170–4173 µs。
c384 で交互 12 回（job **69542**）。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| stride 33 | 3094.4 | 3093.0–3097.9 |
| 2 tile_i 同一 sQ | 4170.2 | 4168.3–4176.8 |

レンジは重ならない。**+34.7%**。戻した。§15.16 と同じく 8 出力で占有が落ちる。
Q 天井 −21% は残るが、この形では取れない。

### 15.68 採用: x を 2 `tile_j` × 2 `tile_i` に組み替える（job 69554）

Q は i に依らない。§15.67 は 4 `j` を残したまま 2 `i` を足して 8 アキュムレータ
にした。今回は 4 `j` をやめて 2 `j` × 2 `i` の 4 アキュムレータのまま、パネルは
1 枚（stride 33）にする。グリッドは 32768/要素のまま。点変化 `Ne=1` / `Ne=2` は
全点ビット一致。login 3-run **8.965 ms / 2950.9 µs**。c182 で交互 12 回（job
**69554**）。分母は §15.63 の xpad。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| stride 33 4j dual sQ | 3088.4 | 3085.2–3094.3 |
| **2j × 2i 同一 sQ** | **2945.2** | 2942.2–2950.5 |

レンジは重ならない。**−4.6%**。採用。Q の充填を 16 `tile_i` から 8 pair に半減し、
アキュムレータは 4 のまま占有を保つ。§15.67 が取れなかった Q 天井の一部。

### 15.69 採用: y の 4 `tile_i` を 4 枚 `sQ` で同時内積（job 69565）

§15.61 で z が +21.9% だったので測らずにいた。y の D は i に依らない。`sQ0`–`sQ3`、
8 アキュムレータ、グリッド 16384/要素。x/z は §15.68 のまま。点変化 `Ne=1` /
`Ne=2` は全点ビット一致。login 3-run **8.900 ms / 2929.0 µs**。c182 で交互 12 回
（job **69565**）。分母は §15.68 の x2j2i。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| 2j × 2i | 2953.2 | 2951.3–2956.2 |
| **y 4 tile_i 4 sQ** | **2929.0** | 2925.8–2933.5 |

レンジは重ならない。**−0.8%**。採用。z の 4-way は 8 アキュムで占有が死んだが、
y は `sD` `double2` のまま D 充填を 4 `tile_i` で共有するので内積負荷が増えても
勝つ。8 出力でも y では屋根が落ちない。

### 15.70 不採用: x の 4 `tile_j` × 2 `tile_i`（job 69573）

§15.61 で測らずにいた x 側。4 `j` × 2 `i`、dual `sQ` stride 33、8 アキュムレータ、
グリッド 16384/要素。y/z は §15.68 のまま（y 4-way は載せない）。点変化は全点
ビット一致。login 12.552–12.563 ms / 4167.8–4171.2 µs。c390 で交互 12 回（job
**69573**）。分母は §15.68 の x2j2i。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| 2j × 2i | 2954.6 | 2951.9–2959.1 |
| x 4j × 2i 8 アキュム | 4173.8 | 4171.5–4181.5 |

レンジは重ならない。**+41.3%**。戻した。§15.67 と同じ占有死。y で 8 出力が勝って
も x の `__ldg` 内積では 8 本のアキュムが屋根を潰す。

### 15.71 現行（y 4-way）の分割アブレーション（job 69599）

分母は同一ジョブの y4q **2929.4 µs**（c390、3-run 中央値）。数値は壊れる。

| 不正変更 | µs/stage | 対 2929.4 | 対 §15.66（3094.1） |
|---|---:|---:|---:|
| INNER=1 | 1564.4 | **−46.6%** | −45.7% |
| D オペランドを 1.0 | 2863.3 | **−2.3%** | −15.2% |
| Q/vel を 1.0 | 2323.8 | **−20.7%** | −20.7% |
| global 全部 1.0 | 1775.2 | **−39.4%** | −31.9% |
| エピローグ省略 | 3112.4 | +6.2% | +15.0% |
| 全バリア削除 | 3292.6 | +12.4% | −0.5% |

x を 2`i` にしたあと **D 天井は −15% から −2.3% に落ちた**。残る契約内の賞金は
Q（−21%）で、INNER は契約外。バリア削除が遅くなるのはコード生成の副作用。

### 15.72 不採用: y の 4 `sQ` を畳まず `double2` に載せる（job 69602）

§15.15 で測らなかった「16 回のまま `double2` だけ」。隣接 2 `i` を
`sQ01`/`sQ23` の `double2` にし、内積回数は 16 のまま。点変化は全点ビット一致。
login 10.090–10.098 ms / 3332.4–3334.7 µs。c390 で交互 12 回（job **69602**）。
分母は §15.69 の y4q。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| y 4 tile_i | 2929.7 | 2927.8–2933.9 |
| sQ `double2` 詰め | 3338.1 | 3335.2–3341.1 |

レンジは重ならない。**+13.9%**。戻した。DFMA は減らず、16 バイト詰めの税だけが
残る。z の同手は同一仮説なのでやらない。

### 15.73 ncu: y 4-way 対 2 `tile_i`（job 69608）

c384、同一ジョブ、`--set full`。採否に使わない ncu 時間。y 以外は不変。

| | x2j2i y | y4q y |
|---|---:|---:|
| Duration | 2155 µs | 2104 µs |
| グリッド | 32768 | 16384 |
| Waves/SM | 13.47 | 6.74 |
| L1/TEX | 98.8% | 98.8% |
| L1 hit | 27.1% | 10.5% |
| DRAM | 4.0% | 4.3% |
| 占有率 | 97.5% | 95.3% |
| レジスタ | 32 | 32 |
| local spill 要求 | 41.5 M | 33.8 M |

両方とも `__launch_bounds__(128, 16)` の 32 レジスタ天井で **y が local にスピル**している。4-way の壁時計 −0.8% はブロック半減による D 充填減で、スピルは残る。x/z は spill 0。次は y のレジスタ予算を上げてスピルを消す。

### 15.74 採用: y だけ `__launch_bounds__(128, 8)`（job 69612）

`minBlocks=16` はスレッドあたり 32 レジスタが上限で、y の 8 アキュムが local に
溢れていた（§15.73）。8 に下げると 64 レジスタまで使える。x/z は 16 のまま。
点変化 `Ne=1` / `Ne=2` は全点ビット一致。login 3-run **7.902 ms / 2590.3 µs**。
c384 で交互 12 回（job **69612**）。分母は §15.69 の y4q。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| y 4-way `minBlocks=16` | 2926.4 | 2924.4–2930.4 |
| **y `minBlocks=8`** | **2596.7** | 2593.4–2603.4 |

レンジは重ならない。**−11.3%**。採用。占有を半分にしてスピルを消す方が、
L1 屋根の y では勝つ。§15.55 の z `minBlocks=8`（+2.6%）は spill が無いカーネル
だった。ncu job **69619**（c384、y4q と ylb8 同一ジョブ）: y の local spill
33.8 M → **0**、レジスタ 32 → 64、理論占有 100% → 50%、ncu Duration 2087 →
1463 µs。x/z は不変。

### 15.75 採用: x の 4 `j` × 2 `i` を `minBlocks=8` で再試（job 69620）

§15.70 は 8 アキュムを 32 レジスタで回して +41.3%。y と同じくレジスタ予算を
64 にして 4-way を載せる。y は §15.74 のまま。点変化は全点ビット一致。
login 3-run **7.272 ms / 2376.0 µs**。c384 で交互 12 回（job **69620**）。
分母は §15.74 の ylb8。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| y `minBlocks=8` | 2595.5 | 2591.0–2598.6 |
| **+ x 4j×2i `minBlocks=8`** | **2375.3** | 2372.5–2378.8 |

レンジは重ならない。**−8.5%**。採用。§15.70 の負けは 4-way そのものではなく
32 レジスタ天井だった。

### 15.76 採用: z の 4 line を `minBlocks=8` で再試（job 69626）

§15.61 は +21.9%。x/y と同じ 64 レジスタ予算で 4 `sQ`・8 アキュム・グリッド
16384。点変化は全点ビット一致。login 3-run **7.205 ms / 2353.6 µs**。c178 で
交互 12 回（job **69626**）。分母は §15.75 の x4lb8。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| x 4j×2i `minBlocks=8` | 2374.5 | 2371.6–2379.8 |
| **+ z 4-line `minBlocks=8`** | **2354.2** | 2347.9–2358.8 |

レンジは重ならない。**−0.9%**。採用。§15.61 の負けもレジスタ不足だった。

### 15.77 不採用: y の `minBlocks` 8→12（job 69632）

64 レジスタと占有 75% の中間。理論上限は 42 レジスタ。点変化は全点ビット一致。
login 7.402–7.415 ms / 2420.7–2425.2 µs。c387 で交互 12 回（job **69632**）。
分母は §15.76 の z4lb8。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| y `minBlocks=8` | 2351.9 | 2347.4–2358.7 |
| y `minBlocks=12` | 2424.6 | 2421.5–2428.6 |

レンジは重ならない。**+3.1%**。戻した。8 アキュムには 64 レジスタが要り、
占有を足すとまた溢れる。

### 15.78 現行（x/y/z 4-way `minBlocks=8`）の分割アブレーション（job 69629）

分母は同一ジョブの z4lb8 **2360.3 µs**（c384、3-run 中央値）。数値は壊れる。

| 不正変更 | µs/stage | 対 2360.3 | 対 §15.71（2929.4） |
|---|---:|---:|---:|
| INNER=1 | 1689.8 | **−28.4%** | −46.6% |
| D オペランドを 1.0 | 2021.1 | **−14.4%** | −2.3% |
| Q/vel を 1.0 | 1887.2 | **−20.0%** | −20.7% |
| global 全部 1.0 | 1620.0 | **−31.4%** | −39.4% |
| エピローグ省略 | 2339.5 | −0.9% | +6.2% |
| 全バリア削除 | 2242.0 | **−5.0%** | +12.4% |

4-way + 64 レジスタのあと INNER 天井は縮み、**D 天井が −2% から −14% に戻った**。
Q は −20% のまま。バリアは誤差ではなくなった。INNER は契約外。

### 15.79 ncu: z 4-way `minBlocks=8` 対 x 4-way のみ（job 69630）

c384、同一ジョブ、`--set full`。採否に使わない ncu 時間。x/y は不変。

| | x4lb8 z | z4lb8 z |
|---|---:|---:|
| Duration | 1739 µs | 1484 µs |
| グリッド | 32768 | 16384 |
| 占有率 | 97.4% | 48.6% |
| レジスタ | 32 | 63 |
| local spill | 0 | 0 |
| L1/TEX | 98.4% | 96.1% |
| DRAM | 34.1% | 40.0% |

y がなお最長（1466 µs、L1 97.6%、spill 0）。y の Q は j に依らないのに 16 個の
`tile_j` が同じ Q を読み直す。次は y を 4 `j` × 2 `i` に組み替えて Q を共有する。

### 15.80 採用: y を 4 `j` × 2 `i`（job 69640）

グリッド 16384 のまま。`k` / `quad_j` / `pair_i` は x と同じ。Q は 2 `i` だけ
shared に載せ、4 `j` が共有する。`sD` は 16×32 で 4 本の D を `double2` 2 本。
点変化 `Ne=1` / `Ne=2` は全点ビット一致。login 3-run **6.689 ms / 2178.9 µs**。
c384 で交互 12 回（job **69640**）。分母は §15.76 の z4lb8。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| z4lb8（y は 4 `i` × 2 `j`） | 2352.7 | 2349.1–2362.3 |
| **+ y 4 `j` × 2 `i`** | **2184.0** | 2177.7–2188.0 |

レンジは重ならない。**−7.2%**。採用。Q 天井 −20% の一部を取った。

### 15.81 採用: z を 4 `k` × 2 line（job 69645）

y と同じ組替え。Q（line と `l` だけに依る）を 4 つの出力 `k` で共有する。
`quad_k` は 8、`pair` は 2048。点変化 `Ne=1` / `Ne=2` は全点ビット一致。
login 3-run **5.800 ms / 1877.2 µs**。c384 で交互 12 回（job **69645**）。
分母は §15.80 の y4j2i。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| y 4 `j` × 2 `i` | 2181.1 | 2174.1–2189.6 |
| **+ z 4 `k` × 2 line** | **1880.8** | 1879.7–1881.7 |

レンジは重ならない。**−13.8%**。採用。z は 16 個の `tile_k` が同じ Q を
読み直していた。

### 15.82 ncu: z 4 `k` × 2 line 対 y4j2i（job 69647）

c384、同一ジョブ、`--set full`。採否に使わない ncu 時間。x/y は不変。

| | y4j2i z | z4k z |
|---|---:|---:|
| Duration | 1482 µs | 1152 µs |
| DRAM | 40.0% | 28.0% |
| L1/TEX | 96.1% | 96.8% |
| L2 hit | — | 21.0% |
| レジスタ | 63 | 64 |
| local spill | 0 | 0 |
| 占有率 | 48.6% | 48.7% |

z の Duration が y と同じ帯（y 1148 µs、x 1080 µs）。3 本とも L1/TEX 屋根。
y の L1 hit は 5.0%（Q は 4 `j` で共有したあとストリーム）。次の契約内は
D 天井と、y/z の sQ 行パディング（x は §15.63 で stride 33 が −1.2%）。

### 15.83 不採用: y の sQ 行ストライド 16→17（job 69650）

x の stride 33（§15.63）の y 版。点変化は全点ビット一致。login 5.801–5.802 ms /
1878.0–1878.2 µs。c384 で交互 12 回（job **69650**）。分母は §15.81 の z4k。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| z4k | 1880.8 | 1879.9–1881.9 |
| y sQ stride 17 | 1880.7 | 1879.7–1881.7 |

レンジは完全に重なる。**差なし（−0.00%）**。戻した。16 幅の内積は x の 32
幅と違ってバンクが既に空いている。z の同手は同一仮説なのでやらない。

### 15.84 現行（z 4 `k`）の分割アブレーション（job 69649）

分母は同一ジョブの z4k **1880.6 µs**（c384、3-run 中央値）。数値は壊れる。

| 不正変更 | µs/stage | 対 1880.6 | 対 §15.78（2360.3） |
|---|---:|---:|---:|
| INNER=1 | 1194.5 | **−36.5%** | −49.4% |
| D オペランドを 1.0 | 1592.1 | **−15.3%** | −32.5% |
| Q/vel を 1.0 | 1668.0 | **−11.3%** | −29.3% |
| global 全部 1.0 | 1373.8 | **−27.0%** | −41.8% |
| エピローグ省略 | 1893.7 | +0.7% | −19.8% |
| 全バリア削除 | 1830.2 | **−2.7%** | −22.5% |

Q 天井は §15.78 の −20% から **−11.3%** に縮んだ（y/z の Q 共有）。D は
**−15.3%** のまま最大の契約内賞金。バリアは −5.0% から −2.7%。INNER は契約外。
エピローグは誤差。

### 15.85 不採用: x の隣接 `i` で D を `double2` `__ldg`（job 69656）

`i0=pair*32+2*tx`, `i1=i0+1` にして `D(i0,l)`/`D(i1,l)` を 16 バイト 1 本にする。
点変化は全点ビット一致。login 5.812–5.814 ms / 1881.4–1882.0 µs。c384 で交互
12 回（job **69656**）。分母は §15.81 の z4k。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| z4k（i は +16） | 1884.5 | 1883.8–1885.9 |
| D `double2` 隣接 i | 1889.2 | 1888.2–1889.8 |

レンジは重ならない。**+0.25%**。戻した。D 本数を半分にしても L1 屋根では
16 バイトのほうが高く、stride 16 の 2 本 `__ldg` の方が安い。

### 15.86 不採用: y の D を shared ではなく `__ldg`（job 69661）

4-way 64 レジスタのあとで §15.8 の y 版を再試。sD を消し内積で 4 本 `__ldg`。
点変化は全点ビット一致。login 6.47–6.48 ms / 2104.9–2107.7 µs。c178 で交互
12 回（job **69661**）。分母は z4k。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| z4k（y は sD `double2`） | 1879.1 | 1878.4–1880.5 |
| y D `__ldg` | 2108.1 | 2106.7–2112.0 |

レンジは重ならない。**+12.2%**。戻した。4-way でも y の D は stride 256 の
global より shared の方が安い。z は同一仮説なのでやらない。

### 15.87 カーネル別 D=1 アブレーション（job 69662）

数値は壊れる。分母は同一ジョブの z4k **1878.5 µs**（c178）。

| 不正変更 | µs/stage | 対 1878.5 |
|---|---:|---:|
| x の D だけ 1.0 | 1650.0 | **−12.2%** |
| y の D だけ 1.0 | 1840.8 | −2.0% |
| z の D だけ 1.0 | 1860.8 | −1.0% |

全カーネル D=1 の −15.3%（§15.84）の大半は **x の `__ldg` 内積**。次は x の
D タイルをレジスタに載せる（`minBlocks=4` で 128 レジスタ）。

### 15.88 不採用: x の D タイルをレジスタ常駐 + `minBlocks=4`（job 69666）

32 本の `d0`/`d1` をタイル充填後にレジスタへ載せ、内積は L1 を叩かない。
`__launch_bounds__(128, 4)`。点変化は全点ビット一致。login 5.897–5.902 ms /
1910.6–1912.1 µs。c384 で交互 12 回（job **69666**）。分母は z4k。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| z4k | 1879.0 | 1877.4–1880.6 |
| x D レジスタ + `minBlocks=4` | 1912.5 | 1911.8–1912.8 |

レンジは重ならない。**+1.8%**。戻した。占有 50%→25% の税が D の L1 削減を上回る。

### 15.89 採用: 最終タイルの末尾バリア省略（job 69670）

x は 8 タイル、y/z は 16 タイルの最後の `__syncthreads` を外す。点変化は全点
ビット一致。login 3-run **5.791 ms / 1874.2 µs**。c384 で交互 12 回（job
**69670**）。分母は z4k。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| z4k | 1879.3 | 1878.1–1880.8 |
| **末尾バリア省略** | **1874.9** | 1874.0–1875.7 |

レンジは重ならない。**−0.23%**。採用。全削除天井 −2.7% のうち最終 1 本分。

### 15.90 カーネル別 Q=1 アブレーション（job 69672）

数値は壊れる。分母は nsync **1874.5 µs**（c178）。

| 不正変更 | µs/stage | 対 1874.5 |
|---|---:|---:|
| x の Q だけ 1.0 | 1815.4 | −3.2% |
| y の Q だけ 1.0 | 1799.2 | −4.0% |
| z の Q だけ 1.0 | 1804.1 | −3.8% |

Q 天井 −11.3% は 3 本に分かれる。どれか 1 本を潰しても 4% 級。

### 15.91 不採用: y を 8 `j` × 1 `i`（job 69834）

Q を 1 パネルにして 8 本の j が共有する。sD は 16×64。点変化は全点ビット一致。
login 6.149–6.154 ms / 1995.7–1997.4 µs。c103 で交互 12 回（job **69834**）。
分母は nsync。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| nsync | 1879.3 | 1878.2–1881.2 |
| y 8 `j` × 1 `i` | 2001.1 | 1999.6–2002.2 |

レンジは重ならない。**+6.5%**。戻した。Q 半減より sD 64 幅の充填の方が高い。
z の 8 `k` × 1 line は同一仮説なのでやらない。

### 15.92 採用: y の sQ/sD 二重バッファ（job 69918）

充填を次タイルのバッファへ回し、タイルあたりバリアを 2 本から 1 本にする。
点変化は全点ビット一致。login 3-run **5.774 ms / 1868.6 µs**。c384 で交互
12 回（job **69918**）。分母は nsync。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| nsync | 1874.6 | 1874.1–1875.1 |
| **y 二重バッファ** | **1869.3** | 1868.2–1870.3 |

レンジは重ならない。**−0.28%**。採用。残バリア天井の一部。次は z に同じ手。

### 15.93 不採用: z の sQ/sD 二重バッファ（job 69920）

y と同じ充填/計算の 1 バリア化。点変化は全点ビット一致。login 3-run
**5.888 ms / 1907.9 µs**。c178 で交互 12 回（job **69920**）。分母は ydb。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| ydb | 1869.8 | 1868.3–1871.0 |
| z 二重バッファ | 1908.3 | 1906.9–1910.9 |

レンジは重ならない。**+2.1%**。戻した。z の shared は y より大きく、二重化が
占有か充填帯域を食う。次は x（sQ のみ、D は `__ldg`）に同じ手。

### 15.94 不採用: x の sQ 二重バッファ（job 69927）

y と同じ 1 バリア化。D は `__ldg` のまま。点変化は全点ビット一致。login 3-run
**5.835 ms / 1889.9 µs**。c182 で交互 12 回（job **69927**）。分母は ydb。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| ydb | 1864.6 | 1864.0–1866.2 |
| x 二重バッファ | 1887.9 | 1886.8–1888.8 |

レンジは重ならない。**+1.2%**。戻した。x はタイルが 8 と少なく、sQ を 2 面に
すると shared が約 8.4 KB → 16.9 KB になる。次は現行 x の D を shared タイルへ
（§15.8 の逆。当時は 1 点/スレッドで `__ldg` が勝った）。

### 15.95 不採用: x の D タイルを shared 32×32（job 70030）

ブロックが持つ 32 本の `i` × 32 `l` を shared に載せ、内積の D を LDS にする。
点変化は全点ビット一致。login 3-run **5.863 ms / 1898.9 µs**。c179 で交互
12 回（job **70030**）。分母は ydb。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| ydb | 1873.2 | 1871.9–1874.5 |
| x D shared 32×32 | 1904.5 | 1903.2–1905.3 |

レンジは重ならない。**+1.7%**。戻した。§15.8 と同じ結論が 4-way でも立つ。
次は x の D `__ldg` を 1 段ソフトウェアパイプラインする。

### 15.96 不採用: x の D `__ldg` 1 段パイプライン（job 70035）

内積の次 `t` の D を先に発行する。点変化は全点ビット一致。login 3-run
**5.771 ms / 1867.9 µs**。c384 で交互 12 回（job **70035**）。分母は ydb。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| ydb | 1869.1 | 1867.9–1870.2 |
| x D パイプライン | 1868.8 | 1867.3–1870.6 |

レンジは重なる。**差が無い**。戻した。コンパイラが既に LDG を内積と重ねている。
次は y エピローグの Escale を `__ldg` にする（x は既にそうしている）。

### 15.97 不採用: y エピローグの Escale `__ldg`（job 70039）

点変化は全点ビット一致。login 3-run **5.770 ms / 1867.3 µs**。c384 で交互
12 回（job **70039**）。分母は ydb。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| ydb | 1869.1 | 1868.3–1870.1 |
| y Escale `__ldg` | 1868.2 | 1867.1–1869.3 |

レンジは重なる。**差が無い**（12 対中 11 で新が速いが 0.05%）。戻した。
次は z エピローグの Escale `__ldg`。

### 15.98 不採用: z エピローグの Escale `__ldg`（job 70040）

点変化は全点ビット一致。login 3-run **5.774 ms / 1868.5 µs**。c178 で交互
12 回（job **70040**）。分母は ydb。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| ydb | 1869.7 | 1869.0–1871.7 |
| z Escale `__ldg` | 1870.4 | 1869.0–1872.6 |

レンジは重なる。**差が無い**。戻した。エピローグは残天井ではない。
次は x を 4 `i` × 2 `j` に（D 再利用を増やし Q を減らす。4j×2i の逆）。

### 15.99 不採用: x を 4 `i` × 2 `j`（job 70044）

D を 4 本 `__ldg` し Q パネルを 1 面にする。点変化は全点ビット一致。login 3-run
**18.06 ms / 6036 µs**。c384 で交互 12 回（job **70044**）。分母は ydb。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| ydb | 1868.9 | 1868.1–1870.2 |
| x 4 `i` × 2 `j` | 6038.4 | 6007.9–6057.5 |

レンジは重ならない。**+223%**。戻した。x の D `__ldg` を倍にすると L1 が壊れる。
y/z の 2-outer × 4-inner は同一仮説なのでやらない。次は x の D を `__ldcg`
（L1 を Q に残す）。

### 15.100 不採用: x の D を `__ldcg`（job 70271）

読み専用キャッシュではなく L2 経路にして L1 を Q に残す。点変化は全点ビット一致。
login 3-run **5.788 ms / 1873.7 µs**。c387 で交互 12 回（job **70271**）。分母は ydb。

| | 中央値 µs/stage | レンジ |
|---|---:|---:|
| ydb | 1871.6 | 1870.1–1872.6 |
| x D `__ldcg` | 1876.8 | 1875.9–1878.7 |

レンジは重ならない。**+0.28%**。戻した。D は L1 に居た方が速い。

### 15.101 契約内の残りは測り尽くした

本番は §15.92 の ydb のまま。login **5.774 ms / 1868.6 µs**。TC / FUSED =
918.9 / 1868.6 = **2.04×**。

この節以降に測った契約内ノブ（すべて不採用）:

| 節 | 内容 | 結果 |
|---|---|---|
| 15.93 | z sQ/sD 二重バッファ | +2.1% |
| 15.94 | x sQ 二重バッファ | +1.2% |
| 15.95 | x D shared 32×32 | +1.7% |
| 15.96 | x D `__ldg` パイプライン | 差なし |
| 15.97 | y Escale `__ldg` | 差なし |
| 15.98 | z Escale `__ldg` | 差なし |
| 15.99 | x 4 `i` × 2 `j` | +223% |
| 15.100 | x D `__ldcg` | +0.28% |

残天井（分母は時点によるアブレーション）:

- **INNER=1（契約外）**: 全体 −36.5%（§15.84）。長さ 256 の内積そのもの。
- **D=1**: 全体 −15.3%、うち **x −12.2%**（§15.87）。`__ldg`・shared・32 レジスタ
  `minBlocks=4`・隣接 `double2`・パイプライン・`__ldcg` はいずれも負けか差なし。
- **Q=1**: x −3.2% / y −4.0% / z −3.8%（§15.90）。8j と 4i×2j は負け。
- **バリア全削除**: −2.7%（§15.84）。末尾省略（§15.89）と y 二重バッファ（§15.92）
  で一部を取った。x/z 二重バッファは負け。

y/z の 2-outer×4-inner は §15.99 と同一仮説。`__ldcs` / `__ldca` は §15.100 と
同一のキャッシュヒント族。INNER を短くする書き換えは契約外。

これ以上、契約を保ったまま測れるノブは残っていない。

## 16. p=255 `FUSED_TC` の残り天井（2026-08-30）

commit `265ef15`、login node GPU 1、`make CUDA=1 GPUFLAGS=-gpu=cc100`、
入力 `namelists/perf_p255_fused_tc.conf`（`Ne=1`、`nstep=20`、graph off）。
µs/stage は `CUDA device fused tendency` を 19 ステップ × 3 RK = 57 で割った値。
採否の A/B は同一 GPU で変種を交互に走らせた。ncu ジョブ `70946` は ncu 用 conf
を計算ノードの `/tmp` に置いたため起動前に落ちた。機構は §6 の最終形
（`math_pipe` + `wait`、どの資源も単独律速ではない）を分母にする。

ベースライン（5-run 中央値）: Main **2.958 ms/step**、device **912.6 µs/stage**。
README の 918.9 µs は 2026-08-29 の別セッションで、表は残す。

§9 が残していた staging 冗長・占有率・`BK=32` と、他次数から移植できる
`elembnd` 重ねを測った。採用ゼロ。

| 候補 | device 中央値 | 対 baseline | 機構 |
|---|---:|---:|---|
| 64×64, 4×4, 128 thr, `MINB=3`, `BK=16` 二重バッファ（現行） | 912.6 µs | — | — |
| **64×128 タイル、二重バッファのまま**（256 thr, `MINB=1`） | 1150 µs | **+26%** | n タイルを 2 枚にまとめて flux A の再読を半分にする手。shared 48 KB、占有率 18.75%→12.5%。§7 の単一バッファ大タイル負けがパイプライン後も同じ |
| **`BK=32` 単一バッファ**（shared 32 KB, `MINB=3`） | 1247 µs | **+37%** | チャンク 16→8、バリア半減。プリフェッチが mma と重ならない。§4.1 の二重バッファが無いと `BK` を広げても負け |
| `D1D` の `__ldg` | 912.1 µs | ±0 | L2 に既に載っている。レンジ重複 |
| `elembnd` 全消し（不正、天井） | 890.4 µs | **−2.44%**（22.3 µs） | 面カーネルは fused 計時に含まれる。GEMM 経路で隠せる 20 µs 級 |
| 面 2,4 を x の前、面 1,3,5,6 を side stream で x と重ね | 916.5 µs | **+0.44%** | 隠せるのは 22 µs の 2/3 だが、range ローンチ 5 本の代金が上回る。x は DRAM 12% でも発行は埋まっている |
| `launch_bounds(128,2)` | 1002 µs | **+9.8%** | レジスタ上限 256 でコンパイラが占有率を 2 ブロックに落とす。`MINB=3` が上限 |

staging 18.6% の契約内の取り方はタイルを広げることだけで、二重バッファ後に
それをやると占有率で §3 と同じ負け方をする。エピローグ 9.7% は `Escale` と
`dqdt` の RMW で、x+y 融合は §7 どおりレジスタ不足のまま。`elembnd` の 22.3 µs
は実装するとローンチ増で消える。残る天井は契約外（速度の代表スカラー、
呼び出し間で速度が不変、という仮定）か、既に測って負けた占有率操作である。

**当時はここで探索終了と書いたが、§17 で compact 2 ローンチの重ねと
faces 2,4 の x 融合が残っていた。** 表の不採用行はそのまま証拠である。

## 17. compact 面重ねと faces 2,4 の x 融合（2026-08-30、採用）

親は `265ef15`。login node GPU 1、`namelists/perf_p255_fused_tc.conf`。
µs/stage は device fused を 57 で割る。A/B は同一 GPU で交互。タイルは
§16 どおり 64×64 / `BK=16` / 128 thr / `MINB=3` の静的 32 KB 二重バッファ。
`FUSED_DFMA` は同一ドライバ（iso-schedule）。

§16 の range ローンチ重ねは +0.44% で落ちたが、隠せる 22 µs のうち
faces 2,4 は x の入力なので重ねられない。残る 4 面を **2 本の compact
group ローンチ**（group 0 = 面 2+4、group 1 = 面 1,3,5,6）にまとめると
ローンチ代金が消え、group 1 を side stream で x と重ねられる。

| 候補 | device 中央値 | 対直前 | 採否 |
|---|---:|---:|---|
| §16 ベースライン | 912.6 µs | — | — |
| **2 compact group + group 1 を x と重ね** | 909.0 µs | **−0.36%** | 採用。点変化 `dqdt` は Ne=1/Ne=2 で ovl 前と全点ビット一致 |
| `BK=32` 二重バッファ（動的 64 KB, `MINB=2`） | 〜1010 µs | **+11%** | 不採用。チャンク半減より占有率 |
| 128×128、512 thr、`MINB=1`、動的 64 KB | 〜1010 µs | **+11%** | 不採用。§7/§16 の大タイル負けの続き |
| group 0 を消す（不正、天井） | 891.6 µs | ovl から **−1.9%**（17 µs） | 面 2,4 は x の臨界パスに残る |
| **faces 2,4 を x エピローグ前に 128 スレッドで評価**（group 0 ローンチ削除。死んだ `sA[0]` に 128 本） | **901.9 µs** | ovl から **−0.77%** | **採用。** 点変化 `dqdt` は ovl と Ne=1（16,777,216 点）/ Ne=2（33,554,432 点）で全点ビット一致 |
| DIR=0 パネルの `double4` ロード | 903.8 µs | f24 から **+0.21%** | 不採用。発行幅よりレジスタ。`double4` は CUDA 13 で deprecated |
| group 1 を消す（不正、天井） | 900.2 µs | f24 から **−0.18%**（1.7 µs） | 残り 4 面は x と既に重なっている。y へ融合する賞金が誤差級。p=31 では同手が +25% |

採用後: Main **2.927 ms/step**、device **901.9 µs/stage**（login GPU 1、
4-run 中央値 5.1405×10⁻² s / 57）。README 横断表の 918.9 µs は 2026-08-29
の別セッションのまま残す。

faces 2,4 の 17 µs のうち x に移して消えたのは約 7 µs。残りは x の
gather そのもので、専用カーネルを消しても演算は残る。group 1 の 1.7 µs
は契約内で取りに行く値ではない。

横展開: compact 2 ローンチ重ねは、面を別カーネルに出している
`FUSED_TC` が p=255 以外に無いので移植先が無い。faces 2,4 の x 内評価は
p=31 が 4 形とも負けており（`p31_gap_study.md` §18–19）、Nq=256 で
ブロックが 64 個の (j,k) を 128 スレッドで一対一に持つ形に限って効いた。

## 18. faces 2,4 の M 側を体積アドレスにする（2026-08-30、採用）

親は §17 の作業ツリー。login GPU 1、`namelists/perf_p255_fused_tc.conf`。
A/B は同一 GPU で交互。`FUSED_DFMA` は同一ソース。

§17 のあと faces 2,4 の gather が x に ~10 µs 残っていた。M 側は Fmask どおり
owned の `i=Nq-1`（面 2）と `i=0`（面 4）なので `VMapM` 経由の従属ロードは
冗長である。P 側は隣要素／ハローなので `VMapP` のまま。

| 候補 | device 中央値 | 対 f24 | 採否 |
|---|---:|---:|---|
| §17 f24 | 901.9 µs | — | — |
| 面 2,4 の gather を MMA の前へ（レジスタに保持） | 902.6 µs | **+0.08%** | 不採用。16 チャンクをまたぐ live レジスタ |
| **M 側を `eo+(i,j,k)` 直アドレス、P 側は `VMapP`** | **899.3 µs** | **−0.27%** | **採用。** 点変化 `dqdt` は f24 と Ne=1/Ne=2 で全点ビット一致 |
| P 側も M に置き換える（不正、天井） | 893.2 µs | mvol から **−0.68%**（6.1 µs） | P 側 gather が残りの本体 |
| P 側アドレスを MMA 前に `prefetch.global.L2` | 899.0 µs | mvol とレンジ重複 | 不採用。CC の §15.10 と同じで先読みはロードを残したまま |

採用後: Main **2.921 ms/step**、device **899.3 µs/stage**（4-run 中央値
5.1260×10⁻² s / 57）。P 側 6.1 µs は契約上消せない。前倒しと L2 prefetch
は両方測って取れない。

## 19. x の Escale を面 2,4 gather の下に置く（2026-08-30、採用）

login GPU 1。Escale ロードを 1.0 に置き換える不正アブレーションは
5.058×10⁻² s、mvol から **−1.35%（12.1 µs）**。消すのは契約外なので、
x 分だけ面 2,4 の P gather と重ねる。

| 候補 | device 中央値 | 対 mvol | 採否 |
|---|---:|---:|---|
| §18 mvol | 899.3 µs | — | — |
| Escale を 1.0 にする（不正、天井） | 887.4 µs | **−1.35%**（12.1 µs） | 契約外 |
| **x の Escale 16 対を面 gather の前に発行** | **895.1 µs** | **−0.47%** | **採用。** 点変化 `dqdt` は mvol と Ne=1/Ne=2 で全点ビット一致 |
| y/z の Escale を最終 MMA の ISSUE スロットへ | 1031 µs | **+15%** | 不採用。最終チャンクの mma とオペランドポートが衝突 |

採用後: Main **2.905 ms/step**、device **895.1 µs/stage**（4-run 中央値
5.1023×10⁻² s / 57）。x で隠れたのは天井 12.1 µs のうち約 4 µs。y/z の
残りは同じ手口では取れない。

## 20. y/z エピローグの `dqdt` / Escale を先に発行する（2026-08-30、採用）

login GPU 1。y/z の `dqdt` RMW ロードを消す不正アブレーションは
5.068×10⁻² s、esov から **−0.66%（6.0 µs）**。ロードを消すのではなく、
Lift / flux の前に 16 対まとめて発行するとその待ちが他のエピローグ仕事の
下に入る。

| 候補 | device 中央値 | 対直前 | 採否 |
|---|---:|---:|---|
| §19 esov | 895.1 µs | — | — |
| y/z の `dqdt` ロードを消す（不正、天井） | 889.0 µs | **−0.66%**（6.0 µs） | 契約外 |
| **y/z の `dqdt` 16 対をエピローグ先頭で発行** | **886.2 µs** | **−0.99%** | **採用。** 天井より大きい（他ロードと重なる）。点変化は esov と Ne=1/2 全点ビット一致 |
| **同じブロックで y/z の Escale も先読み** | **883.2 µs** | dqh から **−0.33%** | **採用。** 点変化は dqh と Ne=1/2 全点ビット一致 |

採用後: Main **2.871 ms/step**、device **883.2 µs/stage**（4-run 中央値
5.0345×10⁻² s / 57）。y/z を最終 MMA に載せた §19 の +15% とは違い、
mma が終わったあとのエピローグ内ハイストである。

## 21. P 側 gather を x Escale より先に発行する（2026-08-30、占有 GPU で不採用）

login GPU 1。y/z の `flux_bnd` を 0 にする不正アブレーションは
**+0.37%**（5.052 vs 5.033）。ロードを消すと他の待ちが露出し、先読みの
賞金は無い。P 側 6.1 µs は、Escale の前に gather を出すと x の Escale
発行とその待ちが重なる。

| 候補 | device 中央値 | 対 esy | 採否 |
|---|---:|---:|---|
| §20 esy | 883.2 µs | — | — |
| y/z `flux_bnd` を 0（不正、天井） | 886.3 µs | **+0.37%** | 消すと遅くなる。先読み対象にしない |
| P 側 gather を x Escale より先に発行 | 882.1 µs | −0.12% | login 暫定。点変化は esy と Ne=1/2 全点ビット一致。占有 GPU の確認では差なし（下記） |

login 暫定値: Main **2.869 ms/step**、device **882.1 µs/stage**（4-run 中央値
5.0280×10⁻² s / 57）。

2026-09-01、commit は引き続き親 `265ef15` の作業ツリー、Slurm job **73982**、
node **c178**、入力 `namelists/perf_p255_fused_tc.conf`。凍結 `esy` と `p1st` を
同一割当で順序を交互にして 12 組測った。

| | device 中央値 | 全レンジ | Main 中央値 |
|---|---:|---:|---:|
| esy（§20） | **882.442 µs/stage** | 881.688–884.204 | 2.870110 ms/step |
| P gather 先行 | **882.452 µs/stage** | 881.111–883.437 | 2.869105 ms/step |

全レンジが重なり、非対応中央値差は **+0.001%**、候補が速いのは 7/12 組。
paired 差中央値 −0.058% もレンジに対して十分小さい。したがって**差なしで不採用**、
ソースは §20 の esy に戻した。login の −0.12% は共有 GPU の揺らぎだった。

## 22. P 側隠蔽とエピローグ発行順の残り（2026-08-31、採用ゼロ）

候補の分母は §21 の作業ツリー（凍結 `scale-dg_extraction.p255tc_p1st`）。login GPU 1、
`namelists/perf_p255_fused_tc.conf`。µs/stage は device fused / 57。A/B は同一
GPU で変種を交互 4 回。点変化 `dqdt` は試した候補すべて Ne=1/Ne=2 で p1st と
全点ビット一致。`FUSED_DFMA` は同一ソース。ptxas は
`tendency_p255_kernel` が **168 レジスタ・32 KB smem・spill 無し**。
65536/168/128 = 3.04 なので `MINB=3` がレジスタ屋根。`MINB=4` は
`launch_bounds(128,2)` の +9.8%（§16）と同じく spill 方向で、測らず閉じる。

P 側天井は §18 の 6.1 µs。x Escale の前に出した §21 が隠したのは約 0.7 µs。
残りを shared 経由や最終 mma の空き ISSUE に載せると、ロードは残したまま
ポートとレジスタを奪う。

| 候補 | device 中央値 | 対 p1st | 採否 |
|---|---:|---:|---|
| §21 p1st | 882.1 µs | — | — |
| P 側 4 本を 8 B `cp.async` で死んだ `sA[1]` へ、Escale の下で wait | 891.6 µs | **+1.06%** | 不採用。shared 往復と `wait_group` が、レジスタ LDG が Escale と重なっていた分を直列化する |
| M 側 gather も x Escale より先へ | 882.1 µs | **−0.05%** | 不採用。同一バッチ内レンジは非重複だが差は 0.5 µs 未満。直前の p1st 中央値 882.2 µs が m1st レンジに入る |
| x の Lift1D を face24 バリアの前に発行 | 882.9 µs | **+0.10%** | 不採用。Lift1D は L2 常駐で、バリア待ちの下に載せる仕事が無い |
| y/z の `flux_bnd` をストアループの外へハイスト | 882.2 µs | ±0 | 不採用。レンジ重複。§21 で flux を消すと +0.37% だったので、発行位置を動かしても天井が無い |
| 最終チャンクの空き ISSUE に P 側 gather | 885.5 µs | **+0.35%** | 不採用。§18 の全チャンクハイスト +0.08% と同じで、空きスロットでも最終 mma のオペランドポートと衝突する。§19 の y/z Escale を ISSUE へ載せた +15% の縮小版 |
| P 側 `q,u,v,w` を `__ldg` | 890.0 µs | **+0.87%** | 不採用。散在ハローでも read-only キャッシュは損。§16 の D1D `__ldg` ±0 より悪い |
| group 1（面 1,3,5,6）の M 側を Fmask 直アドレス | 882.0 µs | ±0 | 不採用。レンジ重複。カーネル全体の天井が §17 の −0.18%（1.7 µs）で、M 側の従属ロードを外しても fused 計時に出ない |

候補評価時はいったん p1st に戻したが、§21 の占有 GPU A/B により最終ソースは
esy に戻した。最速は `CUDAFORTRAN_FUSED_TC` のまま。

最終再ビルドは Slurm job **73985**、node **c178** で
`SCALE_DG_VARYING_COEFF=1` とし、凍結 esy に対して Ne=1（16,777,216 点）/ Ne=2
（33,554,432 点）の owned `dqdt` がともに**全点ビット一致**した。
旧 `launch_tendency_xyz_p255_*` export は `flux_bnd` を使う従来形を保持し、
本番 dir export だけが faces 2,4 を融合するよう compile-time template で分けた。
最終バイナリ対凍結 esy の占有 GPU A/B は job **73986**、node **c178**、同じ入力で
12 組。esy **883.660 µs/stage**（881.672–885.275）、最終版
**883.781 µs/stage**（881.911–884.395）、差 **+0.014%**、最終版が速いのは
5/12 組で、全レンジが重なるため**性能同一**。これを現行値とする。

ncu job `73620`（`--set full`、c183）と `73624`（stall 明示メトリクス、c179）。
凍結 p1st、conf は `output/perf_p255_fused_tc_ncu.conf`（job `70946` は login の
`/tmp` conf で落ちた）。x/y/z とも Compute (SM) **77%**、DRAM **13–15%**、
L1/TEX **53–54%**、達成占有率 **18%**（理論 18.75%、168 レジスタ）。stall は
`math_pipe` 3.4–4.0 / `wait` 2.8–3.1 が支配で、`long_scoreboard` は 0.3–0.5。
shared バンクコンフリクトは ld ~5×10⁴ / st ~1×10⁵ に対し shared wavefront は
2.5×10⁷ なので 0.5% 未満。§6 の「どの資源も単独律速ではない、math_pipe + wait」
はエピローグ改修後も同じ。face group は別ストリーム、ncu Duration 12 µs
（クロック固定なので µs/stage の分母にしない）。`long_scoreboard` 23.8 はこの
カーネルだけだが、fused 計時への寄与は skip 天井 1.7 µs が上限。

残る契約内の天井で測定誤差（login ~0.5%）を超えるのは P 側本体の ~5 µs
だけだが、消すのは範囲外で、載せるスロットは上記で全部負けた。group 1 の
1.7 µs も誤差級。§21 の −0.12% は占有 GPU job 73982 で差なしと確定した。

## 追記（2026-09-01）: volume GEMM の y を 3 段にした

`p511_gap_study.md` §12 で `VolumeGemmSet` の y GEMM のパイプライン段数を
4 → 3 に落とした。占有率は動かず（レジスタが CTA 数を決めている）、
**命令数が 0.80% 減る**。これらの GEMM は SM スループット 95% の発行律速で、
4 段目は長いプロローグの代金しか払っていない。利得は `K/TileK = Nq/16` 回の
ループにその固定費を薄める形なので、次数が上がるほど小さくなる。

この次数での占有 GPU 交互 A/B は次のとおり（詳細と job 番号は
`p511_gap_study.md` §12.4）。過去の節の数値は当時値としてそのまま残す。

| 経路 | 効果 |
|---|---:|
| `CUDAFORTRAN_GEMM_FUSED` (Nq=256) | **−0.603%** |
| `CUDAFORTRAN_GEMM_CUTE` (Nq=256) | **−0.271%** |

`GEMM_CUTE` の `dqdt` は `CUDAFORTRAN_GEMM` とビット一致、`GEMM_FUSED` は
最大絶対差 1.78e-15（点変化係数、16,777,216 点）で、いずれも変更前と同じ。

## 23. p=255 CC の 3 カーネルに `__restrict__`（2026-09-01、不採用 = 差なし）

commit `4189e3d`（`feature/cuda`）、GB200 1 GPU、`make CUDA=1 GPUFLAGS=-gpu=cc100`。
入力は新設の `namelists/perf_p255_fused.conf`（`perf_p255_fused_tc.conf` の
`DqdtKernel_Type` だけを `CUDAFORTRAN_FUSED` にしたもの。`Ne=1`、`nstep=20`、
graph off）。µs/stage は `CUDA device fused tendency` ÷ 57（19 ステップ × 3 RK）。
採否の A/B は Slurm で占有した GPU（job **75757**、ノード c182）で 2 バイナリを
交互に 12 回。

### 23.1 動機と前提

`cuda_dg_kernels_fused_highp.cu` のうち、p=255 用の 3 本
（`tendency_x/y/z_p255_cc_kernel`）だけが引数ポインタに `__restrict__` を
持っていなかった。同ファイルの p=15/31/63/127、`cuda_dg_kernels_fused.cu` の
p=7、`cuda_dg_kernels_tc.cu` の TC 版はすべて `const double *__restrict__` 形式で、
`p15_gap_study.md` §16.10 のとおり `__restrict__` は次数をまたいで効いている
（`FUSED_TC` で p=15 −2.2%、p=31 −4.0%、p=63 −11.6%、p=127 −1.2%、p=255 −5.7%。
p=7 だけは 32 レジスタ・8 ブロック/SM の設計が spill して +3.6% と逆効果）。

**この節は §15.21 の再測定である。** §15.21（job `68028`）は当時のカーネル
（3547 µs/stage）で `__restrict__` 単独を測り、中央値 −0.05%・レンジ重複で
「差なし」としていた。その後 §15.58–15.100 で x/y/z は 4-way dual `sQ`、
`launch_bounds(128,8)`、y の sQ/sD 二重バッファへと作り替えられて 1868 µs/stage
になっており、当時の「32 レジスタ固定でスケジューリング余地が無い」という
説明はもう前提が違う。だから測り直した。

### 23.2 レジスタと spill（`nvcc -arch=sm_100 -Xptxas -v`）

| カーネル | 変更前 | 変更後 |
|---|---|---|
| `tendency_x_p255_cc_kernel` | 64 reg / spill 0 / smem 8448 B | 64 reg / spill 0 / smem 8448 B |
| `tendency_y_p255_cc_kernel` | 64 reg / spill 0 / smem 16384 B | 64 reg / spill 0 / smem 16384 B |
| `tendency_z_p255_cc_kernel` | 64 reg / spill 0 / smem 8192 B | 64 reg / spill 0 / smem 8192 B |

`__launch_bounds__(128, 8)` は 1024 スレッド/SM ＝ 65536/1024 = **64 レジスタ**が
上限で、3 本ともちょうど上限に張り付いている。p=7 で `__restrict__` が負けた
機構（レジスタ増 → spill）は**ここでは起きない**: ptxas の割り当ても spill も
shared も 1 バイトも動かない。

SASS は変わる。`cuobjdump -sass` で y は `LDG.E.64` 52 本のうち 44 本、
z は 40 本のうち 32 本が `LDG.E.64.CONSTANT`（非コヒーレント経路）に変わり、
x も `LDG.E.64.CONSTANT` が 981 → 1009 本に増える。つまり `__restrict__` は
効いてはいるが、効き目はロードのキャッシュ経路の付け替えだけで、
スケジューリング（レジスタ割り当て・命令数）は動いていない。

### 23.3 数値検証

`SCALE_DG_VARYING_COEFF=1`、`SCALE_DG_DUMP_DQDT` で owned `dqdt(:,1:Ne)` 全点。

| 比較 | Ne=1（16,777,216 点） | Ne=2（33,554,432 点） |
|---|---|---|
| 変更前 CC 対 変更後 CC | **ビット一致** | **ビット一致** |
| `CUDAFORTRAN_GEMM` 対 変更後 CC | 最大絶対差 3.55e-15（相対 4.40e-16） | 最大絶対差 3.55e-15（相対 4.40e-16） |

入力は `namelists/val_p255_fused.conf` / `val_p255_gemm.conf` と、その
`NeX` だけを 2 にした一時コピー（コミットしない）。

### 23.4 A/B（job `75757`、c182、交互 12 回）

| | 中央値 µs/stage | レンジ |
|---|---:|---|
| 現行（`__restrict__` 無し） | 1869.9 | 1868.2–1872.4 |
| `__restrict__` | 1868.1 | 1866.8–1869.1 |

中央値 **−0.098%**。レンジが重なるので **差が無い**。同じジョブの他タイマも
同符号・同オーダーで、いずれもレンジ重複:

| タイマ | 現行 中央値 (レンジ) | `__restrict__` 中央値 (レンジ) | 差 |
|---|---|---|---:|
| `Main per step` [ms] | 5.7760 (5.7710–5.7857) | 5.7713 (5.7672–5.7747) | −0.082% |
| `Step loop per stage` [µs] | 1924.7 (1923.0–1927.9) | 1923.2 (1921.8–1924.3) | −0.080% |

**戻した。**

### 23.5 なぜ効かないか

p=255 CC の 3 本は §15.87 / §15.90 の D=1 / Q=1 アブレーションのとおり
global ロードが天井の 15.3% / 3.2–4.0% しか持っておらず、本体は長さ 256 の
内積そのもの（INNER=1 で −36.5%）である。`__restrict__` が与えるのは
(a) エイリアス仮定を外したスケジューリング余地と (b) 非コヒーレント経路の
ロードだが、(a) はレジスタが 64 の天井に張り付いているので ptxas が使えず
（割り当ても命令数も不変）、(b) は §15.53/15.54/15.56/15.97/15.98 で
`__ldg` を個別に当てて全部「差なし」と出ている道と同じである。
p=15/31/63/127 の CC/TC カーネルで `__restrict__` が二桁 % 効いたのは、
そちらが global ロード律速（`long scoreboard` が支配）だったからで、
p=255 CC はその律速ではない。§15.21 の結論はカーネルを作り替えた後も
成り立つ ── ただし理由は「32 レジスタ固定」ではなく「64 レジスタの天井に
張り付いていて、かつロード律速ではない」である。

p=255 CC の本番コードは §15.92 のまま、1869.9 µs/stage（占有 GPU 中央値）。

## 24. x volume GEMM の `64x64` batched 化を Nq=256 で測る（2026-09-01、不採用 +0.22%）

対象は `CUDAFORTRAN_GEMM_FUSED`（と共有ゲートで `CUDAFORTRAN_GEMM_CUTE`）。
親 `c386c85`、GPU は RIKYU GB200 1 枚、Slurm job `75947`（占有ノード c398）。
入力は `namelists/perf_p255_fused.conf` の `DqdtKernel_Type` だけを
`CUDAFORTRAN_GEMM_FUSED` に差し替えたもの（`Ne=1`, `PolyOrder=255`,
`nstep=20`, `UseCudaGraph=.false.`）。12 回交互 A/B。

### 24.1 §10.4 の batched x と何が違うか

`p575_gap_study.md` §11.2 は `Nq>=512` の x を **y と同じ `64x64` batched**
（`GemmYScaleShallow`、3 段、`run_gemm_batched_nn_scaled` の capped grid.z、
epilogue padding 8、バッチ全体を 1 ローンチ）にして p=575 で **−2.94%** を得た。
本節はそのゲート `Nq >= 512` を `Nq >= 256` に下げ、Nq=256 でだけ符号を見る。

§10.4 の「x GEMM を y と同じ batched 形（A=D1D stride 0）に → +0.7%」は
**4 段の `GemmYScale` を使った別の形**で、capped grid.z も 1 ローンチ化も
入っていない。つまり本節は「同じ結論の再測」ではなく、`Nq>=512` で勝った
現行の形を Nq=256 に下ろす初めての測定である。
（`p767_gap_study.md` 側の記録どおり、Nq が上がるほど効き目は薄くなる方向で、
p=767 / 1023 では §11.16 の差も 0.25% / 0.18% まで縮んでいる。）

実装は `cuda_cutlass_gemm_fused.cu` の 2 つのゲート（融合 x の
`run_volume_gemm_x_scale` と `GEMM_CUTE` の `run_volume_gemm_x`）を
1 つの定数で同時に動かした。**片方だけ動かすと経路役割違反になる**ので、
両方が同じ mainloop を取ることを保った。

### 24.2 結果

| | Main/step | `Step loop`/stage | `CUDA device GEMM fused` | `FUSED volume GEMM only` |
|---|---:|---:|---:|---:|
| A ゲート `Nq>=512`（現行） | 3.05205e-3 | **1.01680e-3** | 5.38338e-2 | 4.59755e-2 |
| B ゲート `Nq>=256` | 3.05879e-3 | **1.01901e-3** | 5.39731e-2 | 4.61281e-2 |
| B − A | +0.221% | **+0.217%** | +0.259% | **+0.332%** |

レンジ（`Step loop`/stage、12 回）は A 1.01418e-3..1.01882e-3、
B 1.01632e-3..1.02149e-3 で**重なる**。ただし**4 つの指標すべてで符号が同じ**で、
x だけを含む `FUSED volume GEMM only` で差が最も大きい（+0.332%）。
判定は**不採用**。ゲートは `Nq >= 512` のまま。

### 24.3 機構

`FUSED volume GEMM only` の +0.33% は x 単独の劣化としては +1% 前後にあたり、
§10.4 の +0.7% と同じ向き・同じ桁である。x は Nq=256 では
`m=Nq=256, n=nq2*Ne=65536, k=256` の 1 本の GEMM で、`64x128` タイルは
n 方向に 512 タイル取れて既に SM を埋めている。`64x64` batched に割ると
同じ FLOP を 2 倍の CTA 数で回すので、CTA あたりの K ループが同じまま
プロローグ／エピローグの回数だけが増える。`Nq>=512` で勝ったのは
そこでの x が `64x128` で **SM 84.5%** しか出ていなかったからで
（`p575_gap_study.md` §11.2）、Nq=256 の x は §10.5 の ncu で既に **84%**
台の mainloop にいて、y（86%）との差が p=575 ほど開いていない。
**タイルを y に寄せる余地そのものが Nq=256 には無い**というのが結論である。

### 24.4 数値

`SCALE_DG_VARYING_COEFF=1`、`dqdt(:,1:Ne)` 全点。

| 比較 | 点数 | max abs |
|---|---:|---:|
| B `GEMM_FUSED` vs A `GEMM_FUSED`（`Ne=1`） | 16,777,216 | **ビット一致** |
| B `GEMM_FUSED` vs `CUDAFORTRAN_GEMM` | 16,777,216 | 1.776e-15（相対 2.219e-16） |
| B `GEMM_CUTE` vs A `GEMM_CUTE` | 16,777,216 | **ビット一致** |
| B `GEMM_CUTE` vs `CUDAFORTRAN_GEMM` | 16,777,216 | 0 |
| B `GEMM_FUSED` vs A `GEMM_FUSED`（`Ne=2`） | 33,554,432 | **ビット一致** |
| B `GEMM_FUSED` vs `CUDAFORTRAN_GEMM`（`Ne=2`） | 33,554,432 | 1.776e-15 |

Nq=256 では A も B も同じ和の順序なので、ビット一致は期待どおりである。
コードは戻した（作業ツリーに残らない）。

## 25. `FUSED_DFMA` を現行ソースで測り直す（2026-09-02、機構比 A 2.278×）

`TODO.md` §2.3。手順・入力の作り方・iso-schedule であることの証拠・6 次数の
一覧は [`p63_gap_study.md`](p63_gap_study.md) §55。本節は p=255 の数値だけを残す。

- commit `fcf1872` + §55.1 の defect 修正（p=63 / p=127 のみに効く 2 行）
- job `78053`、node `c384`、GB200 1 GPU
- 入力 `namelists/perf_p255_fused_tc.conf`（`Ne=1`、`nstep=20`、計時 19 ステップ）
- 12 ラウンド、`FUSED_TC` → `FUSED_DFMA` → `FUSED` の順で交互
- 物差し: `CUDA device fused tendency` ÷ (19 steps × 3)

| 経路 | device 中央値 [ms] | device min–max [ms] | **µs/stage** | `Step loop per stage` [µs] |
|---|---:|---:|---:|---:|
| `CUDAFORTRAN_FUSED_TC` | 50.462 | 50.384–50.522 | **885.30** | 959.10 |
| `CUDAFORTRAN_FUSED_DFMA` | 114.969 | 114.822–115.029 | **2017.00** | 2071.56 |
| `CUDAFORTRAN_FUSED` | 106.888 | 106.844–106.939 | **1875.23** | 1930.48 |

3 経路のレンジは互いに重ならない。

- **機構比 A = 2017.00 / 885.30 = 2.278×**（旧 1998.0 / 883.8 = 2.261×）。
  `Step loop` では 2071.56 / 959.10 = 2.160×。
- 主比 B = 1875.23 / 885.30 = **2.118×**（§15.92 の 1868.6 / §22 の 883.8 で
  2.11×。0.35% 差で一致）。

**`FUSED_DFMA` は 1998.0 → 2017.00 µs/stage（+0.95%）と、6 次数で唯一
「遅くなった」側である。** §17–§22（compact 重ね、面 2,4、M 側直アドレス、
エピローグ先読み）は `FUSED_TC` を 912.6 → 883.8 にしたが、DFMA では利得が
出ていない。**ただし旧値は別ジョブ・別ノードの測定**なので、この 1% を
「TC 専用に効いた」と断定はしない。次に p=255 の `FUSED_TC` を触るときに、
DFMA を同じジョブに入れて確かめる（`TODO.md` §6 の
「`p255_gap_study.md` §22 の探索終了宣言の再点検」と同じ機会で足りる）。

もし本当なら機構としては筋が通る: §17–§22 の 4 つはいずれも
**mma のオペランドを待たせない**ための変更（LDG の本数、面点の再利用、
エピローグの先読み）であり、内積が DFMA になって演算側が 2.3 倍長くなれば、
隠すべきレイテンシの相対量が減って利得も消える。ncu で確かめるなら
`long_scoreboard` と `lg_throttle` を TC / DFMA の両方で 1 ジョブに入れる。

点変化係数の owned `dqdt` 16,777,216 点（`namelists/val_p255_gemm.conf`、
`Ne=1`）は `FUSED_TC` と `FUSED_DFMA` で**ビット一致**、`CUDAFORTRAN_GEMM`
対照で両者とも max abs **3.55e-15**（相対 4.40e-16）。p=255 に
`CUDAFORTRAN_SPLIT` は無いので参照は `GEMM` を採った。

`AGENTS.md` が求める **`Ne>1` のスモークも通した**: `val_p255_gemm.conf` の
`NeX/NeY/NeZ` を 2 にした作業コピー（`Ne=2³`、**134,217,728 点**）で
`SCALE_DG_VARYING_COEFF=1` の owned `dqdt(:,1:Ne)` を出すと、
**`FUSED_TC` / `FUSED_DFMA` / `CUDAFORTRAN_FUSED`（CC）の 3 経路が全点ビット一致**
（max abs 0.0）。`Ne=1` の内部タイル分解が `Ne>1` でも同じ結果を出すことの確認。
