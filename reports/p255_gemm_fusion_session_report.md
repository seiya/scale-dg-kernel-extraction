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
