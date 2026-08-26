# CUTLASS volume GEMM の MMA 命令形状（8x8x4 / 16x8x4 / 16x8x8 / 16x8x16）

- 日付: 2026-08-26
- 対象: `CUDAFORTRAN_GEMM_CUTE` / `CUDAFORTRAN_GEMM_FUSED` の volume GEMM
- 測定環境: RIKYU NVIDIA GB200 1 GPU（login node で直接実行）、
  `make CUDA=1 GPUFLAGS=-gpu=cc100`、nvcc は `-arch=native`（= `sm_100`）。
  H100 側は TSUBAME job `8502531`（§8）
- 測定した commit: 本レポートを追加したコミット（親 `9eed9e5`）と、
  K-deep 命令のイテレータを足したその次のコミット
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

**H100（`sm_90`）では逆に大きく効く。**同じコードを TSUBAME で測ると、volume
GEMM だけの device 時間が `GEMM_CUTE` で **2.895 → 2.222 秒（−23%）**になる。
ただし効いたのは cuBLAS が選ぶ **16x8x8 ではなく 16x8x4** で、H100 の最速は
`GEMM_FUSED` + `16x8x4`（Main 5.711 秒、8x8x4 比 −7.2%、cuBLAS 比 −2.1%）
だった。詳細は §8。

---

## 3. 実装した選択機構

namelist `CutlassMmaShape` を追加した（既定 `"8x8x4"`）。

| 値 | InstructionShape | ArchTag | tile K | 状態 |
|---|---|---|---:|---|
| `8x8x4` | `GemmShape<8,8,4>` | `Sm80` | 16 | 既定 |
| `16x8x4` | `GemmShape<16,8,4>` | `Sm90` | 16 | 素の CUTLASS で通る |
| `16x8x8` | `GemmShape<16,8,8>` | `Sm90` | 16 | `cutlass_f64_kdeep_mma.h` が要る（§5） |
| `16x8x16` | `GemmShape<16,8,16>` | `Sm90` | 32 | 同上。加えてタイルが別（§4） |

4 形状とも参照（cuBLAS）と**一致**する（§5.3）。

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

## 5. K が 4 より深い命令は素の CUTLASS 2.x では動かない

### 5.1 最初の測定（修正前）

`SCALE_DG_VARYING_COEFF=1`、p=255 `Ne=1`、owned `dqdt` 全 16,777,216 点を
`CUDAFORTRAN_GEMM`（cuBLAS）とフル比較すると、参照の maxabs が 8.548 に対して

| 形状 | maxabs 差（修正前） |
|---|---:|
| 8x8x4 | 0.000e+00 |
| 16x8x4 | 0.000e+00 |
| 16x8x8 | **6.339e+02** |
| 16x8x16 | **7.364e+02** |

**kK = 4 の形状だけが通る**という結果だった。

### 5.2 原因と修正

CUTLASS 2.x の 64bit 用 warp tile iterator
(`gemm/warp/mma_tensor_op_tile_iterator_sm80.h`) は **4 深の K グループ**を前提に
書かれている。

- Congruous 版（`:86`）は `Policy::Delta = PitchLinearShape<8,4>` で
  `access_idx = c + s * Iterations::kContiguous`。`c` が M/N のチャンク、
  `s` が K グループなので、**K が外側**に並ぶ。atom ごとに K が連続していて
  ほしい `mma.sync.m16n8k8` の operand 順序と食い違う。
- Crosswise 版（`:841`）は順序自体は K が内側で合っているが、末尾の
  64bit 入れ替え（`k_group_idx_ & 1`）と `operator++` の `byte_offset_ ^= 0x40`
  が、**1 fragment = 1 個の 4 深グループ**という前提で書かれている。
  kK=8 なら 1 fragment が 2 グループにまたがるので、片方は誤ったパリティで
  読まれる。

kK=4 ではグループが 1 つしかないので、どちらの問題も起きない。だから
8x8x4 と 16x8x4 は通り、16x8x8 と 16x8x16 は壊れる。

修正は `cutlass_f64_kdeep_mma.h`（新規）に置いた。swizzle を解き直すのではなく、
**実績のある kK=4 のイテレータを kGroups 回まわして、atom ごとに連結する**:

```
KDeepMultiplicandTileIterator
  Base = 同じ warp tile iterator を GroupInstructionShape（kK=4）で実体化
  load()          : Base のコピーを kGroups 回 load / ++ し、
                    dst[(atom * kGroups + g) * kRegsPerAtom + r]
                      = src[atom * kRegsPerAtom + r]
  operator++      : Base を kGroups 回進める
  set_kgroup_index: Base には k_group * kGroups を渡す
  add_tile_offset : K 方向の成分を kGroups 倍する
```

これがそのまま広い命令の operand 順序になる。f64 の `m16n8k8` の A は
`m16n8k4` の A を K 方向に 2 つ積んだものであり（`cute/atom/mma_traits_sm90.hpp:67`
の `ALayout` の value 側 stride が `(_8,_64)` = (M 半分, K 半分)）、B も同様、
C は 3 形状で同一である。したがって epilogue には手を入れていない。

差し込みは `cutlass::gemm::warp::DefaultMmaTensorOp` を
（double、`GemmShape<16,8,8>` / `<16,8,16>`、`OpMultiplyAdd`）に対して特殊化し、
`Type` を上のイテレータを使う `MmaTensorOpKDeep`（stock の `MmaTensorOp` から
派生して IteratorA/B だけ差し替えたもの）にするだけである。`DefaultMmaCore`
から下は自動的にこれを拾うので、`cuda_cutlass_gemm_fused.cu` 側は
ヘッダを 1 行 include する以外そのままでよい。

**SASS は増えていない。** 並べ替えはレジスタの割り当てが変わるだけで、
DMMA 本数も LDS 本数も修正前と同じである（§6 の表）。

### 5.3 修正後の数値検証

`Ne=1` と `Ne=2`（スモーク）の両方、`GEMM_CUTE` と `GEMM_FUSED` の両方、
4 形状すべてについて owned `dqdt` を全点比較した。

| 経路 | 8x8x4 | 16x8x4 | 16x8x8 | 16x8x16 |
|---|---:|---:|---:|---:|
| `GEMM_CUTE`（`Ne=1`, `Ne=2`） | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `GEMM_FUSED`（`Ne=1`, `Ne=2`） | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 |

`GEMM_FUSED` の 3.553e-15 は融合 epilogue の加算順序による既存の差で、
8x8x4 のときと同じ値である。

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

## 7. GB200 での性能（p=255 `Ne=1`, `nstep=1000`, 3 回の中央値）

`Main` はホスト wall、`device` は `Volume derivate + surface lift` の
CUDA event device 時間。3 回の測定はどれも ±0.3% 以内。**4 形状とも数値は
一致している**ので、そのまま比較してよい。

| 経路 | 形状 | Main [s] | device [s] | 8x8x4 比 |
|---|---|---:|---:|---:|
| `GEMM_CUTE` | 8x8x4 | 3.7401 | 3.7028 | 1.000 |
| `GEMM_CUTE` | 16x8x4 | 3.7388 | 3.7003 | **0.999** |
| `GEMM_CUTE` | 16x8x8 | 3.7694 | 3.7306 | 1.008 |
| `GEMM_CUTE` | 16x8x16 | 5.4156 | 5.3688 | 1.450 |
| `GEMM_FUSED` | 8x8x4 | 3.2329 | 3.1981 | 1.000 |
| `GEMM_FUSED` | 16x8x4 | 3.2307 | 3.1961 | **0.999** |
| `GEMM_FUSED` | 16x8x8 | 3.2894 | 3.2550 | 1.018 |
| `GEMM_FUSED` | 16x8x16 | 7.5262 | 7.4754 | 2.338 |

- **16x8x4 は 8x8x4 と同じ**（差 0.1%、測定ばらつきの範囲）。§6 のとおり
  SASS が同一なのだから当然である。
- 16x8x8 が 0.8〜1.8% 遅いのは、PTX 1 命令が 4 個の `DMMA.8x8x4` に展開される
  ときのレジスタ配置の都合と思われる（DMMA 本数も LDS 本数も同じ）。
- 16x8x16 が大きく遅いのは §4 のタイル（K=32、shared 倍、occupancy 半分）で、
  命令形状の効果ではない。

命令形状で SASS が変わらない以上、nsys / ncu によるカーネル単位の分解は
行っていない。分解すべき差が存在しない。

---

## 8. H100 での測定（TSUBAME、job `8502531`）

- GPU: NVIDIA H100 94 GB（`gpu_1=1`、host `r19n2`、driver 580.105.08、cc 9.0）
- commit `f5794b7`、`nvfortran 26.1-0`、
  `make CUDA=1 GPUFLAGS=-gpu=cc90 NVCCFLAGS='... -arch=sm_90 ...'`
- 入力は GB200 と同じ p=255 `Ne=1` `nstep=1000`、`UseCudaGraph = .false.`
- 投入は `job_tsubame_mma_shape.sh`（未コミットのジョブスクリプト）

### 8.1 数値検証（H100 実機の DMMA に対して）

GB200 での検証は ptxas が `DMMA.8x8x4` へ展開した実装に対するものだったので、
本物の `DMMA.16x8x4 / 16x8x8 / 16x8x16` が走る H100 でも同じ比較をやり直した
（`SCALE_DG_VARYING_COEFF=1`、owned `dqdt` 全 16,777,216 点、参照は cuBLAS）。

| 経路 | 8x8x4 | 16x8x4 | 16x8x8 | 16x8x16 |
|---|---:|---:|---:|---:|
| `GEMM_CUTE` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `GEMM_FUSED` | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 |

**GB200 と同じ値**である。`cutlass_f64_kdeep_mma.h` の operand 順序は
命令そのものに対して正しく、`sm_100` での展開に依存していない。

### 8.2 測定

`Main` はホスト wall、`tendency` は tendency 全体の CUDA event device 時間、
`volume GEMM` は volume GEMM だけの device 時間
（`GEMM_FUSED` では融合 z epilogue を含む）。

| 経路 | 形状 | Main [s] | tendency [s] | volume GEMM [s] | 8x8x4 比 |
|---|---|---:|---:|---:|---:|
| cuBLAS（`CUDAFORTRAN_GEMM`） | — | 5.8324 | 4.9612 | — | — |
| `GEMM_CUTE` | 8x8x4 | 6.9163 | 6.0110 | 2.8952 | 1.000 |
| `GEMM_CUTE` | **16x8x4** | 6.1152 | 5.3388 | **2.2222** | **0.768** |
| `GEMM_CUTE` | 16x8x8 | 6.3530 | 5.5762 | 2.4592 | 0.849 |
| `GEMM_CUTE` | 16x8x16 | 8.3999 | 7.6219 | 4.5038 | 1.556 |
| `GEMM_FUSED` | 8x8x4 | 6.1548 | 5.3902 | 4.0945 | 1.000 |
| `GEMM_FUSED` | **16x8x4** | **5.7114** | **4.9048** | **3.6094** | **0.882** |
| `GEMM_FUSED` | 16x8x8 | 6.1386 | 5.3735 | 4.0779 | 0.996 |
| `GEMM_FUSED` | 16x8x16 | 12.7465 | 11.9644 | 10.6690 | 2.606 |

### 8.3 読み取り

1. **H100 では命令形状が効く。** GB200 では 4 形状が 0.1% 以内に並んだのに対し、
   ここでは volume GEMM だけで最大 −23%。§6 の SASS の違いがそのまま出ている。
2. **効くのは 16x8x4 で、cuBLAS が選ぶ 16x8x8 ではない。**
   `GEMM_CUTE` で 16x8x4 が 0.768、16x8x8 が 0.849。DMMA の本数は 16x8x8 の方が
   半分なのに遅い。GB200 でも 16x8x8 だけが 0.8〜1.8% 遅かったので、K を深くする
   側に固有のコスト（K-deep イテレータの並べ替え、レジスタ圧、operand 供給）が
   あると見るのが自然だが、**まだ切り分けていない**（§9）。
3. **H100 の最速は `GEMM_FUSED` + `16x8x4`。** Main 6.1548 → 5.7114 秒（−7.2%）で、
   cuBLAS の 5.8324 秒も下回る。**H100 では既定を 16x8x4 にする価値がある**
   （GB200 では同着なので、既定を変えても損はしない）。
4. `GEMM_FUSED` の比（0.882）が `GEMM_CUTE`（0.768）より 1 に近いのは、
   この timer が帯域律速の融合 z epilogue を含むからで、GEMM 本体の改善が
   薄まって見えているだけである。
5. 16x8x16 は H100 でも大きく損。§4 の K=32 タイル（shared 倍・occupancy 半分）が
   命令形状の利得を完全に食い潰す。

### 8.4 このジョブで採れなかったもの

- **カーネル単位の内訳（nsys）。** ジョブスクリプトの不具合で、cuBLAS 用の
  プロファイル実行に `CutlassMmaShape = "cublas"` という不正な値を書き込んで
  しまい、nsys 段の 1 本目で停止した（`nsys` 自身は成功を返すので気付きにくい）。
  スクリプトは修正済み（ラベルと namelist 値を分離、`ERROR STOP` の検出、
  `TIMING=0` で計測段を飛ばせる）。時間の表は別段階なので影響を受けていない。
  （§8.1 の検証結果は job の標準出力にしか出ていなかった。修正版は
  `validation.txt` にも残す。）

---

## 9. 残っていること

- **H100 で 16x8x8 が 16x8x4 に負ける理由の切り分け。** nsys でカーネル単位に
  分けたうえで、`ncu` でレジスタ数・occupancy・stall 内訳を見る。候補は
  K-deep イテレータの並べ替えコスト、operand レジスタの増加
  （A が 2 → 4 本、B が 1 → 2 本）、`MmaMultistage` の k ループが
  4 → 2 回に減ることによる待ちの露出。
- **既定値をどうするか。** H100 で 16x8x4 が明確に速く、GB200 で同着である以上、
  既定を `16x8x4` にする案がある。ただし `arch::Sm90` タグを既定にすることに
  なるので、sm_80 以前で組む可能性を考えるならビルド時に分岐が要る。
- 16x8x16 は K=32 タイルが必須である以上、このタイル構成では見込みが無い。
