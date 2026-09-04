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
GEMM だけの device 時間が `GEMM_CUTE` で **2.907 → 2.225 秒（−23%）**になる。
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

## 7. GB200 での性能（p=255 `Ne=1`, `nstep=1000`）

**訂正**: 初出のこの表は `Volume derivate + surface lift`（区間の end-to-end
wall 時間）を「device 時間」と書いていた。H100 の表（§8、CUDA event）と
突き合わせられないので、device event で取り直した。結論は変わらない。

`Main` はホスト wall、`tendency` と `volume GEMM` は CUDA event の device 時間
（`GEMM_FUSED` の volume GEMM は融合 z epilogue を含む）。2 回の測定は
±0.1% 以内だったので中央値。**4 形状とも数値は一致している**（§5.3）。

| 経路 | 形状 | Main [s] | tendency [s] | volume GEMM [s] | 8x8x4 比 |
|---|---|---:|---:|---:|---:|
| cuBLAS（`CUDAFORTRAN_GEMM`） | — | 3.7924 | 3.5233 | — | — |
| `GEMM_CUTE` | 8x8x4 | 3.7440 | 3.4757 | 2.2958 | 1.000 |
| `GEMM_CUTE` | 16x8x4 | 3.7425 | 3.4749 | 2.2952 | **1.000** |
| `GEMM_CUTE` | 16x8x8 | 3.7720 | 3.5046 | 2.3240 | 1.012 |
| `GEMM_CUTE` | 16x8x16 | 5.4176 | 5.1492 | 3.9698 | 1.729 |
| `GEMM_FUSED` | 8x8x4 | 3.2347 | 2.9672 | 2.5567 | 1.000 |
| `GEMM_FUSED` | 16x8x4 | 3.2321 | 2.9640 | 2.5524 | **0.998** |
| `GEMM_FUSED` | 16x8x8 | 3.2916 | 3.0223 | 2.6107 | 1.021 |
| `GEMM_FUSED` | 16x8x16 | 7.5204 | 7.2510 | 6.8393 | 2.675 |

- **16x8x4 は 8x8x4 と同じ**（差 0.03〜0.17%、測定ばらつきの範囲）。§6 のとおり
  SASS が同一なのだから当然である。
- 16x8x8 が 1.2〜2.1% 遅いのは、PTX 1 命令が 4 個の `DMMA.8x8x4` に展開される
  ときのレジスタ配置と、K-deep イテレータが足す命令の分である（§8.6）。
- 16x8x16 が大きく遅いのは §4 のタイル（K=32、shared 倍、occupancy 半分）で、
  命令形状の効果ではない。

命令形状で SASS が変わらない以上、GB200 では nsys / ncu によるカーネル単位の
分解は行っていない。分解すべき差が存在しない（H100 では行った。§8.4、§8.6）。

---

## 8. H100 での測定（TSUBAME、job `8502531` / `8502578` / `8502700`）

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
   半分なのに遅い。カーネル単位に割ると、**この負けは x GEMM 1 本に集中している**
   （§8.4）。
3. **H100 の最速は `GEMM_FUSED` + `16x8x4`。** Main 6.1548 → 5.7114 秒（−7.2%）で、
   cuBLAS の 5.8324 秒も下回る。**H100 では既定を 16x8x4 にする価値がある**
   （GB200 では同着なので、既定を変えても損はしない）。
4. `GEMM_FUSED` の比（0.882）が `GEMM_CUTE`（0.768）より 1 に近いのは、
   この timer が帯域律速の融合 z epilogue を含むからで、GEMM 本体の改善が
   薄まって見えているだけである。
5. 16x8x16 は H100 でも大きく損。§4 の K=32 タイル（shared 倍・occupancy 半分）が
   命令形状の利得を完全に食い潰す。

### 8.4 カーネル単位の内訳（job `8502578`、`nstep=20`、60 launch / カーネル）

同じ H100 で nsys を採り直した（`8502531` の nsys 段はスクリプトの不具合で
落ちていた。原因と修正は §8.5）。数値検証と時間は `8502531` と同じ結果で、
`GEMM_CUTE` の volume GEMM 比は 0.765 / 0.845 / 1.547 と再現している。

平均時間 [µs/launch]。x は素の `Gemm`（M=256, N=65536, K=256）、
y は batched（256×256×256、batch 256）、z は batched（65536×256×256、batch 1）。
cuBLAS の列はタイルが違う（自分で選んだもの、grid 形状から役割を同定）。

| GEMM | 自前タイル | 8x8x4 | 16x8x4 | 16x8x8 | 16x8x16 † | cuBLAS（タイル） |
|---|---|---:|---:|---:|---:|---:|
| x | 64x128x16 | 341.3 | **279.8** | 352.4 | 717.7 | 183.7（128x64x16） |
| y | 64x64x16 | 306.1 | **209.4** | 216.0 | 310.4 | 169.8（64x64x16） |
| z | 64x32x16 | 314.5 | **241.8** | 242.2 | 469.4 | 220.2（64x128x16） |
| 合計（×60） | | 57.71 ms | **43.86 ms** | 48.64 ms | 89.85 ms | 34.42 ms |

† 16x8x16 は K=32 タイル（§4）。

**16x8x8 の負けは x GEMM 1 本に集中している。** y と z では 16x8x4 とほぼ同じ
（216.0 対 209.4、242.2 対 241.8）なのに、x だけ 352.4 対 279.8 で、8x8x4 の
341.3 よりも遅い。x は 3 本のうち唯一 warp タイルが 32x64x16 で（他は
32x32x16）、accumulator だけでスレッドあたり 64 本を占める。そこへ kK を
8 にすると operand fragment が倍（A が 4→8、B が 8→16 doubles）になり、
`MmaMultistage` はそれを二重化して持つ。**レジスタ圧が第一容疑**だが、
これは ncu で確かめること（§9）。

`GEMM_FUSED` の z（融合 assembly）は epilogue 律速で、命令形状はほとんど効かない
（702.4 → 695.9、16x8x8 では 774.7 に悪化）。`GEMM_FUSED` の改善率が
`GEMM_CUTE` より小さいのはこのためである。

### 8.5 cuBLAS との差はどこまで縮んだか

volume GEMM 3 本の合計で **57.71 → 43.86 ms**、cuBLAS 34.42 ms に対する比は
**1.68 → 1.27 倍**。命令形状だけで差の約 6 割（13.9 ms / 23.3 ms）が埋まった。

残りは mainloop の実装差である。**y GEMM はタイルも warp 数も stage 数も
cuBLAS と同一**（64x64x16 / 2x2x1 / 4 段）でありながら 209.4 対 169.8 で、
なお 1.23 倍ある。x GEMM を cuBLAS と同じ命令（16x8x8）・同じ 4 warp で
比べると 352.4 対 183.7。CUTLASS 2.x の `MmaMultistage` と cuBLAS の
sm90 カーネルは、同じタイル・同じ命令でも別物だということになる。

### 8.6 ncu: x GEMM の 16x8x8 はレジスタが溢れている（job `8502700`）

`--set full` を `GEMM_CUTE` の x/y/z 1 組（launch 3–5）に当てた。

**x GEMM**（tile 64x128x16、warp 32x64x16、4 warp）:

| | 8x8x4 | 16x8x4 | 16x8x8 |
|---|---:|---:|---:|
| duration [µs] | 367.6 | **251.4** | 345.9 |
| レジスタ/スレッド | 228 | 248 | **255（上限）** |
| **Local Memory Spilling Requests** | 0 | 0 | **4.46 MB** |
| local load / store sectors | 0 / 0 | 0 / 0 | 13,643,464 / 6,432,696 |
| 実行命令数 | 68.17 M | **56.24 M** | 65.72 M |
| Compute (SM) throughput | **86.90%** | 65.80% | 46.13% |
| Memory throughput | 43.07% | 62.89% | 47.13% |
| warp cycles / issued inst | 8.53 | **6.86** | 8.31 |
| Block Limit Registers / Shared Mem | 2 / 2 | 2 / 2 | 2 / 2 |
| Theoretical / Achieved occupancy | 12.50 / 12.17% | 12.50 / 12.22% | 12.50 / 12.22% |

**y GEMM**（warp 32x32x16、4 warp）と **z GEMM**（warp 32x32x16、2 warp）:

| | | 8x8x4 | 16x8x4 | 16x8x8 |
|---|---|---:|---:|---:|
| y | duration [µs] | 369.5 | **248.5** | 255.3 |
| y | レジスタ/スレッド | 164 | 188 | 218 |
| y | spill | 0 | 0 | 0 |
| y | 実行命令数 | 84.38 M | **79.07 M** | 90.54 M |
| z | duration [µs] | 378.4 | **280.7** | 283.9 |
| z | レジスタ/スレッド | 156 | 168 | 220 |
| z | spill | 0 | 0 | 0 |
| z | 実行命令数 | 103.92 M | **100.47 M** | 111.02 M |

### 8.7 読み取り

1. **16x8x8 の x GEMM はレジスタ 255 本（上限）に張り付き、4.46 MB を spill する。**
   16x8x4 は 248 本で spill ゼロ。§8.4 の「レジスタ圧が容疑」は当たりで、しかも
   occupancy の低下ではなく **spill そのもの**である（3 形状とも
   theoretical occupancy は 12.5% で同じ）。spill のせいで実行命令数が
   56.2 M → 65.7 M と**増え**、SM throughput は 46% まで落ちている。
2. **8x8x4 の x GEMM は発行律速**（SM 86.9%）。16x8x4 は同じ仕事を
   68.17 M → 56.24 M 命令でこなし、SM を 65.8% に下げて 367.6 → 251.4 µs
   （1.46 倍）になる。**命令数を減らすことが効いている**という当初の想定どおり。
3. **warp タイルが小さい y と z では 16x8x8 も溢れない**（218 / 220 本）。
   それでも 16x8x4 に僅かに負けるのは、実行命令数が増えるからである
   （y で 79.07 M → 90.54 M、+15%）。K-deep イテレータが 4 深グループを
   2 回まわして並べ替える分のアドレス計算・レジスタ移動が乗っている。
   DMMA の本数が半分になっても、それ以外の命令がそれ以上に増えている。
4. したがって **16x8x4 が 3 本とも最良**という結果には、x では spill 回避、
   y/z では余計な命令の少なさ、という別々の理由がある。
5. x の 16x8x4 も 248 本で上限に近い。タイルや epilogue を触るときは、
   ここが 255 に達しないか注意すること。

### 8.8 スクリプトの不具合（`8502531` の nsys 段）

cuBLAS のプロファイル実行に `CutlassMmaShape = "cublas"` という不正値を
書き込んでしまい（ラベルと namelist 値に同じ引数を使っていた）、アプリが
`ERROR STOP` した。`nsys` 自身は成功を返すので `set -e` に掛からない。
修正済み: ラベルと namelist 値を分離、プロファイル実行後に `ERROR STOP` を
検出して停止、`TIMING=0` で計測段を飛ばせる、検証結果を `validation.txt` にも残す。

---

## 9. 残っていること

- **既定値をどうするか。** H100 で 16x8x4 が 3 本とも最良（volume GEMM 合計で
  −24%、Main −7.2%）、GB200 で同着である以上、既定を `16x8x4` にする案がある。
  `arch::Sm90` タグが既定になるので、sm_80 以前で組む可能性を考えるなら
  ビルド時の分岐が要る。
- **16x8x8 を x で使う道は残っている**が、価値は低い。spill を止めるには
  x の warp タイルを 32x32x16 に落とす（64x128 タイルを 8 warp で持つ）
  必要があり、それは命令形状ではなくタイルの実験になる。現状 16x8x4 が
  spill なしで速いので、追う理由がない。
- **残る 1.27 倍。** 命令形状では埋まらない。同一タイル・同一命令でも
  cuBLAS の sm90 カーネルの方が速く（§8.5）、こちらは y GEMM で
  実行命令数 79.07 M・SM 65.8% とまだ余裕がある。次に効くとすれば
  mainloop 側（CUTLASS 3.x / CuTe の pipeline、あるいはタイル選択）である。
- 16x8x16 は K=32 タイルが必須である以上、このタイル構成では見込みが無い。

---

## 追記（2026-09-04、tree `5e507b8`）: ノブが届く範囲を広げた

本稿の測定は `f5794b7` の時点のもので、当時 x volume GEMM は汎用の
`launch_volume_gemm_x`（64x128 タイル）を通っており、命令形状は x/y/z の
3 本すべてに効いていた。**本稿の数値と結論はそのまま有効である。**

その後 `51d56a8`（2026-09-02）で x volume GEMM に次数別の専用タイルが入り、
その `XTile` が `GS<8,8,4>` をハードコードしていたため、`Nq <= 256` では
`CutlassMmaShape` が x に効かなくなっていた。2026-09-04 にこれを実体化した
（`reports/h100_report.md` §7）。**本稿が測った p=255 については、採用タイル
`PHXTile0` が本稿の汎用 64x128 タイルそのものなので、走るカーネルは
当時と同じである**（§8.6 の x GEMM のレジスタ 255 本 / 4.46 MB spill も、
tile 64x128x16 / warp 32x64x16 という記述のまま当たる）。変わったのは
`Nq <= 64` の x タイルと、低次の融合 carrier で形状が選べるようになった点。
