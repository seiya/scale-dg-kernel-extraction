# CUTLASS volume GEMM の MMA 命令形状（8x8x4 / 16x8x4 / 16x8x8 / 16x8x16）

- 日付: 2026-08-26
- 対象: `CUDAFORTRAN_GEMM_CUTE` / `CUDAFORTRAN_GEMM_FUSED` の volume GEMM
- 測定環境: RIKYU NVIDIA GB200 1 GPU（login node で直接実行）、
  `make CUDA=1 GPUFLAGS=-gpu=cc100`、nvcc は `-arch=native`（= `sm_100`）
- 測定した commit: 本レポートを追加したコミット（親 `9eed9e5`）
- 入力: `bench_runs/p255_gemm_cute.conf` / `bench_runs/p255_gemm_fused.conf`
  （p=255、`Ne=1`、`nstep=1000`、`UseCudaGraph` 未設定 = off）
- 数値検証入力: `input_p255_val_gemm*.conf`（`nstep=1`）＋ `SCALE_DG_VARYING_COEFF=1`

数値契約は変更していない。変えたのは GEMM の MMA 命令形状の選択機構だけである。

---

## 1. 動機

CUTLASS 経路は `GemmShape<8,8,4>`（Ampere の `d884` DMMA）を直書きしていた。
一方、TSUBAME（H100）で採った cuBLAS のカーネル名は同じ問題形状に対して
SM90 の f64 DMMA を選んでいる:

```
sm90_xmma_gemm_f64f64_..._tilesize64x128x16_stage3_..._tensor16x8x8_...
sm90_xmma_gemm_f64f64_..._tilesize128x64x16_stage3_..._tensor16x8x8_...
sm90_xmma_gemm_f64f64_..._tilesize64x64x16_stage4_..._tensor16x8x8_...
```
（`tsubame_ch7_8500661/nsys/p255_CUDAFORTRAN_GEMM_cuda_gpu_kern_sum.csv`。
p=7 側の csv には `tensor16x8x16` も出る。）

対して GB200 の cuBLAS は `cutlass_80_tensorop_d884gemm_*`、つまり 8x8x4 を選ぶ。
**同じ問題形状で H100 と GB200 の選択が違う**ので、GB200 で 16x8x8 が本当に
不利なのかを実測で決めることにした。

---

## 2. 結論（先に）

**GB200（`sm_100`）では f64 の MMA 命令形状を変えても得るものは何もない。**
ptxas が `mma.sync.m16n8k4/8/16.f64` を **`DMMA.8x8x4` の 2 / 4 / 8 命令に
展開する**からである。Hopper が持っていた広い FP64 DMMA 命令は Blackwell の
SM には無い。cuBLAS が GB200 で 8x8x4 のカーネルを選ぶのはこのためである。

同じ理由で、H100 では意味がある変更でありうる（TSUBAME で測る価値は残る）。

---

## 3. 実装した選択機構

namelist `CutlassMmaShape` を追加した（既定 `"8x8x4"`）。

| 値 | InstructionShape | ArchTag | tile K | 状態 |
|---|---|---|---:|---|
| `8x8x4` | `GemmShape<8,8,4>` | `Sm80` | 16 | 既定。参照と**ビット一致** |
| `16x8x4` | `GemmShape<16,8,4>` | `Sm90` | 16 | 参照と**ビット一致** |
| `16x8x8` | `GemmShape<16,8,8>` | `Sm90` | 16 | **数値が合わない**（§5）。計測用に残置 |
| `16x8x16` | `GemmShape<16,8,16>` | `Sm90` | 32 | **数値が合わない**。加えてタイルが別（§4） |

threadblock / warp タイルと stage 数は 8x8x4 のものを据え置いた
（x: 64x128x16 / 32x64x16 / 3 段、y: 64x64x16 / 32x32x16 / 4 段、
z: 64x32x16 / 32x32x16 / 4 段）。3 本の GEMM と `GEMM_FUSED` の
z assembly 融合 epilogue が同じ選択に従う。

`main.f90` → `setup_advect3d_eq_setup` → `cuda_cutlass_set_mma_shape` →
`launch_volume_gemm_cute` / `launch_volume_gemm_xy` / `launch_z_gemm_assembly`
の末尾引数、という経路で渡す（`CublasEmulation` と同じ流儀）。
4 形状ぶんのカーネルは 1 バイナリに実体化してあるので、再ビルドなしで比較できる。

---

## 4. `16x8x16` だけタイルが違う理由

`MmaBase`（`mma_base.h:128,132`）は
`kWarpGemmIterations = WarpShape::kK / InstructionShape::kK` が
**2 以上かつ偶数**であることを要求する。warp の kK=16 のままでは 16x8x16 は
1 になり、コンパイルが通らない。そこで 16x8x16 に限り tile K を 32 にした。
その結果 CTA あたりの shared memory は倍になり、occupancy は半分になる。
下の測定で 16x8x16 が大きく遅いのは主にこれである（命令形状の効果ではない）。

---

## 5. 数値検証

`SCALE_DG_VARYING_COEFF=1`、p=255 `Ne=1`、owned `dqdt` 全 16,777,216 点を
`CUDAFORTRAN_GEMM`（cuBLAS）とフル比較した（`SCALE_DG_DUMP_DQDT`）。
参照側の maxabs は 8.548。

| 経路 | 形状 | maxabs 差 |
|---|---|---:|
| `GEMM_CUTE` | 8x8x4 | **0.000e+00** |
| `GEMM_CUTE` | 16x8x4 | **0.000e+00** |
| `GEMM_CUTE` | 16x8x8 | 6.339e+02 |
| `GEMM_CUTE` | 16x8x16 | 7.364e+02 |
| `GEMM_FUSED` | 8x8x4 | 3.553e-15 |
| `GEMM_FUSED` | 16x8x8 | 6.339e+02 |

### なぜ k>4 で壊れるのか

CUTLASS 2.x の 64bit 用 warp tile iterator
(`gemm/warp/mma_tensor_op_tile_iterator_sm80.h:86` ほか) は
`Policy::Delta = PitchLinearShape<8,4>` で、**k を 4 ずつのグループに切って**
`Iterations::kStrided = InstructionShape::kK / 4` 回ロードする。
fragment のインデックスは `c + s * Iterations::kContiguous`、
つまり **M/N の atom が内側、k グループが外側**に並ぶ。
一方 `MmaTensorOp::operator()` は fragment を atom ごとの
`ArchMmaOperator::FragmentA` の配列として reinterpret するので、
atom は k を連続で持っていなければならない。

- `kK = 4`（8x8x4、16x8x4）: k グループは 1 つだけなので順序問題は起きない。
- `kK = 8, 16`（16x8x8、16x8x16）: 2 / 4 グループが atom 間に挟まって並び、
  `mma.sync.m16n8k8` が期待する operand 順序と一致しない。

直すには f64 用の warp レベル演算子（fragment の並べ替え）を自作する必要がある。
**GB200 では §2 のとおり見返りがゼロなので着手しない。**
選択自体は計測のために残し、選ぶと setup で警告を出す。

---

## 6. 決定的な証拠: `sm_90` と `sm_100` の SASS

`mma.sync.aligned.m8n8k4 / m16n8k4 / m16n8k8 / m16n8k16 .f64` を 1 命令ずつ
含むカーネルを 1 つの `.cu` にまとめ、`nvcc -cubin` の SASS を数えた:

| PTX 命令 | `-arch=sm_90` | `-arch=sm_100` |
|---|---|---|
| `mma.sync.m8n8k4.f64` | `DMMA.8x8x4` × 1 | `DMMA.8x8x4` × 1 |
| `mma.sync.m16n8k4.f64` | **`DMMA.16x8x4` × 1** | `DMMA.8x8x4` × **2** |
| `mma.sync.m16n8k8.f64` | **`DMMA.16x8x8` × 1** | `DMMA.8x8x4` × **4** |
| `mma.sync.m16n8k16.f64` | **`DMMA.16x8x16` × 1** | `DMMA.8x8x4` × **8** |

`sm_100a` / `sm_103` / `sm_120` でも同じ（合計 15 個の `DMMA.8x8x4`）。

実カーネルの SASS でも一致する。`cuobjdump -sass cuda_cutlass_gemm_fused.o` の
DMMA はすべて `DMMA.8x8x4` で、同じ仕事量あたりの本数も同じである:

| カーネル種 | 8x8x4 | 16x8x4 | 16x8x8 | 16x8x16（K=32 タイル） |
|---|---:|---:|---:|---:|
| `Gemm`（x） | 128 | 128 | 128 | 256 |
| `GemmBatched`（y, z） | 128 | 128 | 128 | 256 |
| z assembly 融合 | 64 | 64 | 64 | 128 |

（16x8x16 が倍なのは、タイルが K 方向に倍で 1 CTA の仕事が倍だから。）

---

## 7. 性能（p=255 `Ne=1`, `nstep=1000`, 3 回の中央値）

`Main` はホスト wall、`dev` は `Volume derivate + surface lift` の
CUDA event device 時間。3 回の測定はどれも ±0.3% 以内だったので中央値を載せる。

| 経路 | 形状 | Main [s] | device [s] | 8x8x4 比 |
|---|---|---:|---:|---:|
| `GEMM_CUTE` | 8x8x4 | 3.7400 | 3.7013 | 1.000 |
| `GEMM_CUTE` | 16x8x4 | 3.7402 | 3.7012 | **1.000** |
| `GEMM_CUTE` | 16x8x8 †| 3.7583 | 3.7195 | 1.005 |
| `GEMM_CUTE` | 16x8x16 †| 5.3848 | 5.3388 | 1.442 |
| `GEMM_FUSED` | 8x8x4 | 3.2321 | 3.1943 | 1.000 |
| `GEMM_FUSED` | 16x8x4 | 3.2310 | 3.1944 | **1.000** |
| `GEMM_FUSED` | 16x8x8 †| 3.3138 | 3.2767 | 1.026 |
| `GEMM_FUSED` | 16x8x16 †| 7.4703 | 7.4203 | 2.323 |

† 数値が合わない版（§5）。仕事量と発行命令数は正しい版と同じなので、
時間の比較材料としては意味がある。

- **16x8x4 は 8x8x4 と完全に同じ**（差 0.0〜0.1%、測定ばらつきの範囲）。
  §6 のとおり SASS が同一なのだから当然である。
- 16x8x8 が 0.5〜2.6% 遅いのは、PTX 1 命令あたり 4 個の `DMMA.8x8x4` に
  展開される際のレジスタ配置の都合と思われる。得るものは無い。
- 16x8x16 が大きく遅いのは §4 のタイル（K=32、shared 倍、occupancy 半分）。

命令形状で SASS が変わらない以上、nsys / ncu によるカーネル単位の分解は
行っていない。分解すべき差が存在しない。

---

## 8. 残っていること

- **H100 では未測定。** `sm_90` では 16x8x4 / 16x8x8 / 16x8x16 が本物の
  1 命令になるので、TSUBAME で同じ namelist を振れば cuBLAS の選択
  （`tensor16x8x8`）の妥当性をそのまま確かめられる。ただし 16x8x8 を
  使うには先に §5 の operand 順序を直す必要がある。
- **16x8x8 を正しくするには** f64 用の warp レベル演算子を自作して
  fragment を並べ替えるか、CuTe（`SM90_16x8x8_F64F64F64F64_TN`、
  `cute/atom/mma_traits_sm90.hpp:64`）で mainloop を書き直す必要がある。
  GB200 では見返りがゼロなので着手していない。
