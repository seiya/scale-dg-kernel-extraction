# H100（TSUBAME 4）での測定

- 日付: 2026-08-26
- GPU: NVIDIA H100 94 GB（TSUBAME 4、`gpu_1=1`、driver 580.105.08、cc 9.0）
- 比較対象: RIKYU の NVIDIA GB200 1 GPU（cc 10.0）
- ジョブ: `8500661`（経路横断、commit `9eed9e5`）、
  `8502531` / `8502578` / `8502700`（MMA 命令形状、commit `f5794b7`）
- 投入スクリプト: `job_tsubame_ch7.sh`、`job_tsubame_mma_shape.sh`（どちらも未コミット）

このリポジトリの最適化は GB200 を基準にしている。本稿は**同じコード・同じ入力を
H100 で走らせると何が変わるか**の記録である。数値契約は変更していない。

---

## 1. まとめ

1. **絶対性能は GB200 が速い**（経路により 1.3〜5.4 倍）。ただし p=255 の
   volume GEMM に限れば、命令形状を H100 向けに選ぶと **H100 の方が 3% 速い**
   （741.7 対 765.3 µs/call、§4）。
2. **p=7 の最速は両機とも `CUDAFORTRAN_FUSED_TC`。** ただし CUDA core 版に対する
   優位が違う。GB200 では 1.22 倍、H100 では **1.91 倍**。H100 の FP64 Tensor Core
   ピークが CUDA core の 2 倍あるのに対し、GB200 では同じだからである。
3. **p=255 の最速は機種で入れ替わっていた。** GB200 は `CUDAFORTRAN_GEMM_FUSED`、
   H100 は cuBLAS（`CUDAFORTRAN_GEMM`）。ただしこれは 8x8x4 での話で、
   **`CutlassMmaShape = "16x8x4"` を選ぶと H100 でも `GEMM_FUSED` が最速に戻る**
   （1628.8 対 1652.7 µs/call、§4）。
4. **既定値は変えない。** 既定は GB200 の最速構成（`8x8x4`）のまま。H100 で走らせる
   ときだけ namelist で `16x8x4` を選ぶ（§5）。
5. **達成率の性格が違う。** volume GEMM は GB200 でピークの 84.0%（ほぼ飽和）、
   H100 では 51.9% にとどまる。H100 側にはまだ余地があり、それは cuBLAS との
   1.27 倍の差として見えている（§4.3）。

---

## 2. 2 台のピーク値

| | GB200 (cc 10.0) | H100 (cc 9.0) |
|---|---:|---:|
| FP64 CUDA core peak | 40.1 TFLOP/s | 33.5 TFLOP/s |
| FP64 Tensor Core peak | **40.1 TFLOP/s**（CUDA core と同じ） | **66.9 TFLOP/s**（2 倍） |
| HBM peak | 約 7.9 TB/s | 2.396 TB/s |
| f64 MMA 命令 | `DMMA.8x8x4` のみ（広い形は ptxas が展開） | `DMMA.8x8x4 / 16x8x4 / 16x8x8 / 16x8x16` |

GB200 で Tensor Core 用の 2 倍が撤廃されている件は
`overall_summary_report.md` §7.2 に、f64 MMA 命令の違いは
`sm90_mma_shape_survey.md` §6 に根拠がある。**この 2 行が、以下の差のほとんどを
説明する。**

---

## 3. 経路横断（job `8500661`、commit `9eed9e5`）

tendency 1 回あたりの GPU 時間。GB200 側は `overall_summary_report.md` §7 の
「現行」列（同じツリー = `63a4234` + ±x 面ステージング = `9eed9e5`）。
どちらも nsys のカーネル時間から算出し、p=255 の GEMM 系では
side stream に隠れる `elembnd_flux_kernel` を除いている。

| 経路 | GB200 [µs/call] | H100 [µs/call] | H100 / GB200 |
|---|---:|---:|---:|
| p7 `OPENACC_ASIS` | 1011.2 | 2278.0 | 2.25 |
| p7 `OPENACC_SPLIT` | 730.6 | 2004.6 | 2.74 |
| p7 `CUDAFORTRAN_SPLIT` | 715.7 | 1979.7 | 2.77 |
| p7 `CUDAFORTRAN_FUSED` | 323.8 | 1731.4 | 5.35 |
| p7 `CUDAFORTRAN_FUSED_TC` | **264.9** | **907.9** | 3.43 |
| p7 `CUDAFORTRAN_GEMM` | 1670.7 | 2904.3 | 1.74 |
| p255 `CUDAFORTRAN_FUSED` | 4978.9 | 6276.8 | 1.26 |
| p255 `CUDAFORTRAN_FUSED_TC` | 4420.9 | 7620.4 | 1.72 |
| p255 `CUDAFORTRAN_GEMM` | 1143.0 | **1602.1** | 1.40 |
| p255 `CUDAFORTRAN_GEMM_CUTE` | 1138.1 | 1990.1 | 1.75 |
| p255 `CUDAFORTRAN_GEMM_FUSED` | **970.7** | 1769.4 | 1.82 |

読み取り:

- **p=7 は両機とも `FUSED_TC` が最速**だが、意味が違う。GB200 では CUDA core 版
  `FUSED`（323.8）に対して 1.22 倍でしかないのに、H100 では 1731.4 → 907.9 の
  **1.91 倍**。§2 のとおり H100 だけ Tensor Core のピークが 2 倍だからで、
  `tc_paper_survey_2407.09621.md` が GB200 で「TC 版の優位は 1.16 倍」と
  結論した理由がここで裏返る。**Tensor Core 版を維持する価値は H100 の方が高い。**
- **p=7 の帯域律速な経路ほど H100 が不利**（`OPENACC_SPLIT` 2.74 倍、
  `CUDAFORTRAN_SPLIT` 2.77 倍）。HBM が約 3.3 分の 1 なので当然である。
- **p=255 では差が小さい**（1.26〜1.82 倍）。演算律速なので、
  FP64 ピークの比（40.1 対 66.9 = 0.60、ただし到達率が違う）が効く。
- この時点では **H100 の p=255 最速は cuBLAS**（1602.1）で、`GEMM_FUSED`（1769.4）が
  負けていた。GB200 とは順位が逆である。§4 でこれが変わる。

---

## 4. MMA 命令形状（jobs `8502531` / `8502578` / `8502700`、commit `f5794b7`）

CUTLASS 経路の volume GEMM は MMA 命令形状を namelist `CutlassMmaShape` で
選べる（`8x8x4` / `16x8x4` / `16x8x8` / `16x8x16`）。詳細と実装は
[`sm90_mma_shape_survey.md`](sm90_mma_shape_survey.md) にあり、ここでは
H100 側の結論だけを載せる。

### 4.1 GB200 では効かず、H100 では効く

p=255 `Ne=1`、`nstep=1000`、`UseCudaGraph=.false.`。数値はすべて CUDA event の
device 時間を tendency 1 回あたりに直したもの。**4 形状とも参照（cuBLAS）と
一致することを両機で確認済み**（GB200 は `Ne=1` / `Ne=2`、H100 は `Ne=1`）。

volume GEMM のみ [µs/call]:

| 形状 | GB200 | H100 |
|---|---:|---:|
| 8x8x4（既定） | 765.3 | 969.1 |
| **16x8x4** | 765.1 | **741.7** |
| 16x8x8 | 774.7 | 818.8 |
| 16x8x16 | 1323.3 | 1499.1 |

GB200 では 8x8x4 と 16x8x4 が同着（0.03%）。H100 では 16x8x4 が **−23.5%**。
理由は §2 の 4 行目で、GB200 では `mma.sync.m16n8k4/8/16.f64` が
`DMMA.8x8x4` の 2 / 4 / 8 命令に展開されてしまい、命令数が減らないためである。

**16x8x4 の H100 は GB200 より 3.1% 速い**（741.7 対 765.3）。同じコード・同じ
タイルで、H100 の FP64 Tensor Core ピークが 1.67 倍あることがそのまま出ている。

### 4.2 p=255 の順位が GB200 と揃う

tendency 全体 [µs/call]:

| 経路 | GB200 8x8x4 | H100 8x8x4 | H100 16x8x4 |
|---|---:|---:|---:|
| cuBLAS `CUDAFORTRAN_GEMM` | 1174.4 | 1652.7 | 1652.7 |
| `CUDAFORTRAN_GEMM_CUTE` | 1158.6 | 2008.3 | 1781.1 |
| `CUDAFORTRAN_GEMM_FUSED` | **989.0** | 1801.7 | **1628.8** |

8x8x4 では H100 だけ cuBLAS が最速だったが、16x8x4 にすると `GEMM_FUSED` が
cuBLAS を **1.5% 下回り**、GB200 と同じ順位に戻る。

### 4.3 達成効率と、残っている差

volume GEMM の理論仕事量は 1 tendency 呼び出しあたり 25.77 GFLOP
（x/y/z 各 2·M·N·K）。

| | µs/call | 有効 TFLOP/s | 対 FP64 CUDA peak | 対 FP64 Tensor peak |
|---|---:|---:|---:|---:|
| GB200 8x8x4 / 16x8x4 | 765.3 | 33.67 | 84.0% | 84.0%（同じ値） |
| H100 8x8x4 | 969.1 | 26.59 | 79.4% | 39.7% |
| H100 **16x8x4** | 741.7 | **34.74** | **103.7%** | 51.9% |
| H100 16x8x8 | 818.8 | 31.47 | 94.0% | 47.0% |

H100 の 16x8x4 が **CUDA core ピークを超えている**（103.7%）ことが、Tensor Core
経路が実際に効いている直接の証拠である。一方で Tensor ピークに対しては 51.9% で、
**GB200 の 84.0%（ほぼ飽和）と比べて余地が大きい**。その余地は cuBLAS との差として
実在する（volume GEMM 3 本の合計で 43.9 対 34.4 ms = 1.27 倍）。
`sm90_mma_shape_survey.md` §8.5 のとおり、**タイル・warp 数・stage 数・命令まで
揃えても cuBLAS の sm90 カーネルの方が速い**ので、残りは mainloop の実装差である。

### 4.4 H100 固有の落とし穴: x GEMM のレジスタ spill

`16x8x8` は 16x8x4 に負ける。ncu（job `8502700`）によると x GEMM
（tile 64x128x16、warp 32x64x16）でレジスタが 255 本（上限）に張り付き、
**4.46 MB を spill** する。16x8x4 は 248 本で spill ゼロ。occupancy は 3 形状とも
theoretical 12.5% で同じなので、効いているのは spill そのものである。
warp タイルが小さい y / z GEMM では 16x8x8 も溢れない。

x の 16x8x4 も 248 本と上限に近い。**H100 でこのカーネルのタイルや epilogue を
触るときは、レジスタが 255 に達しないか必ず見ること。**

---

## 5. 既定値の方針

**既定は `8x8x4` のまま**とする。既定値は GB200 の最速構成に合わせるという方針で、
GB200 では 16x8x4 との差が測定ばらつきの範囲（0.03%）だから、既定を動かす利得が
無い。H100 で走らせるときは入力に 1 行足す:

```fortran
  CutlassMmaShape = "16x8x4",
```

対象は `CUDAFORTRAN_GEMM_CUTE` と `CUDAFORTRAN_GEMM_FUSED`。他の経路では無視される。
4 形状ぶんのカーネルは 1 バイナリに実体化してあるので、再ビルドは要らない。

H100 向けビルドは:

```bash
module purge && module load nvhpc
make clean
make CUDA=1 GPUFLAGS=-gpu=cc90 \
  NVCCFLAGS='-O3 -std=c++17 -arch=sm_90 -Ithird_party/cutlass/include --expt-relaxed-constexpr'
```

---

## 6. H100 で残っている課題

- **cuBLAS との 1.27 倍**（§4.3）。命令形状では埋まらない。次に効くとすれば
  mainloop 側（CUTLASS 3.x / CuTe の pipeline、あるいはタイル選択）。
  H100 では Tensor ピークに対して 51.9% しか出ていないので、GB200（84.0%）より
  伸びしろは大きい。
- **p=7 の Tensor Core 経路。** H100 では CUDA core 版に対して 1.91 倍と、
  GB200（1.22 倍）よりずっと効く。`tc_paper_survey_2407.09621.md` が GB200 で
  「割に合わない」と判断した最適化のいくつかは、H100 では再評価の価値がある。
- **p=255 `FUSED_TC` は H100 で特に悪い**（7620 µs/call、GB200 比 1.72 倍）。
  手書き Tensor Core 経路が CUTLASS / cuBLAS の multistage mainloop に負ける
  という GB200 での結論は、H100 でより顕著である。
- **16x8x16 は両機とも不可。** `MmaBase` が warp あたり 2 回以上の GEMM を要求する
  ため K=32 タイルが必須で、shared が倍・occupancy が半分になる分を命令形状の
  利得では取り返せない。

---

## 7. 命令形状ノブの実体化（2026-09-04、tree `5e507b8`）

**§1–§6 は commit `9eed9e5` / `f5794b7`（2026-08-26）の記録であり、表は書き換えない。**
以下は、その後 GB200 側でツリーが動いた結果として **H100 を測り直す前に
直さなければならなかった 1 件**と、その修正の検証である。時間の測定は
含まない（TSUBAME 側の再測定はこれから）。

### 7.1 `CutlassMmaShape` は途中から「半分だけ繋がった」ノブになっていた

`§4` の測定当時（`f5794b7`）、x volume GEMM は汎用の
`launch_volume_gemm_x` を通っており、命令形状は x/y/z の 3 本すべてに効いて
いた。**`§4` の数値はその意味でそのまま有効である。**

その後 `51d56a8`（2026-09-02、GEMM ライブラリ割り当てとエピローグ carrier の
測定）で x volume GEMM に**次数別の専用タイル**が入った
（Nq=8→tile 9、16→5、32→7、64→3、128/256→0）。この `XTile` テンプレートは
`GS<8,8,4>` と `Sm80` をハードコードしていたため、`Nq <= 256` では

- `CutlassMmaShape` を変えても **x だけ 8x8x4 のまま**（`GEMM_CUTE` /
  `GEMM_FUSED` の両方で）、
- 融合の x carrier（Nq = 8 / 16 / 32）は `mma_shape != 0` を即座に拒否、
- Nq = 64 の z carrier も採用 package の分岐に形状が無く拒否、

という状態になっていた。低次では `GEMM_CUTE` だけが（x を 8x8x4 に置いたまま）
起動でき、`GEMM_FUSED` は起動できない。**この非対称は AGENTS.md が
`GEMM_CUTE` と `GEMM_FUSED` に禁じている「異なる mainloop / 命令形状」**
そのものなので、この状態の H100 値は公表できない。

### 7.2 実体化した範囲

- `MmaShapeSel<0..3>`（命令形状 → InstructionShape / ArchTag / TileK）を 1 箇所に。
  16x8x16 だけ `TileK = 32` を要求する（`MmaBase` の
  `WarpShape::kK / InstShape::kK >= 2` かつ偶数）ので、タイル専用化のある
  経路では **形状 0〜2 のみ**実体化し、16x8x16 は従来どおり汎用セット限定。
- `XTile` に InstShape / ArchTag をデフォルト引数付きで追加。**既存のタイル別名の
  型は 1 ビットも変わらない**（`static_assert` で担保）。各次数の**採用タイルのみ**を
  形状付きで実体化（`XAdopted8/16/32/64/Hi`）。
- `VolumeGemmSetP7` / `VolumeGemmSetP15` をテンプレート化。従来、非既定形状の
  低次 y/z は汎用 64x64 タイルに落ちており、**タイルと命令の 2 軸が同時に
  動いていた**。これも一軸化された。
- 融合エピローグ: x carrier は Nq = 8（tile 9 / xpkg 5）、16（tile 5 / xpkg 4）、
  32（tile 7 / xpkg 4）の**採用構成のみ**、z carrier は Nq = 64 の **zpkg 5 のみ**を
  形状付きで実体化。`SCALE_DG_XTILE` などの ablation メニュー（weight_mode 4 通り
  × package 8 通り）は 8x8x4 のまま残す。全部を形状ぶん実体化するとビルドが
  倍増し、どの次数も使わない組み合わせのために払うことになる。
- `launch_volume_gemm_x_tiled` に `mma_shape` 引数を追加（Fortran 側の interface と
  呼び出し 2 箇所も）。

### 7.3 検証（GB200、login GPU、`SCALE_DG_VARYING_COEFF=1`、owned `dqdt` 全 16,777,216 点）

p = 7 / 15 / 31 / 63 / 127 / 255 × `GEMM_CUTE` / `GEMM_FUSED` ×
`8x8x4` / `16x8x4` / `16x8x8` の 36 通り。対照は p<=127 が
`CUDAFORTRAN_SPLIT`、p=255 が `CUDAFORTRAN_GEMM`。

| p | `GEMM_CUTE` 3 形状 | `GEMM_FUSED` 3 形状 | 既定形状 変更前バイナリとの比較 |
|---:|---|---|---|
| 7 | max abs 0.000e+00 | rel 2.078e-16 | ビット一致 |
| 15 | 0.000e+00 | 2.078e-16 | ビット一致 |
| 31 | 0.000e+00 | 2.078e-16 | ビット一致 |
| 63 | 0.000e+00 | 4.156e-16 | ビット一致 |
| 127 | 0.000e+00 | 2.078e-16 | ビット一致 |
| 255 | 0.000e+00 | 2.078e-16 | ビット一致 |

**既定（8x8x4）の出力が変更前の実行ファイルとビット一致である**ことが要点で、
型レベルの `static_assert` と合わせて、公表済みの GB200 の値は 1 つも
無効化されない。p >= 511 は `x_mode = 0`（汎用 x）で今回の分岐を通らない。

ビルド確認: cc100（`-gpu=cc100` / `-arch=sm_100`）、**cc90（`-gpu=cc90` /
`-arch=sm_90`、14 分 30 秒）**、非 CUDA（stub）の 3 モードとも成功。
現行ツリーは `NVCCFLAGS` を全指定しなくても `GPUNVCCFLAGS=-arch=sm_90` で
H100 向けにビルドできる（§5 のコマンドはより短く書ける）。

### 7.4 これから測ること

`jobs/tsubame_paths.sh`（未コミット、`jobs/` は gitignore）が
**現行ツリーの経路 × 次数を H100 で測り直す**ジョブである。§3 の表を
p = 7 / 15 / 31 / 63 / 127 / 255 / 511 / 575 / 767 と全生存経路に広げ、
CUTLASS 2 経路は `8x8x4` と `16x8x4` の両方を採る。p = 1023 は payload
180 GiB で H100 94 GB に載らないので対象外、p = 767 は 78 GiB で綱渡り。
入力は GB200 の表と同じ `namelists/perf_p*_gemm.conf` から
`DqdtKernel_Type` だけを差し替えて生成する。
