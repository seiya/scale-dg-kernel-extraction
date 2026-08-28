# p=575 GEMM / GEMM_FUSED 対応

## 1. 結論

`PolyOrder=575`（`Nq=576`）を `CUDAFORTRAN_GEMM` と
`CUDAFORTRAN_GEMM_FUSED` で実行可能にした。既存の cuBLAS / CUTLASS
カーネルは runtime `Nq` で実装済みであり、必要なコード変更は p=575 を
対応済みの2経路だけに制限する dispatch guard の追加である。

点ごとに変化する `u`, `v`, `w`, `Escale`, `normal_fn`, `Fscale` を使い、
両経路の全 owned `dqdt` 191,102,976 点を比較した。最大絶対差は
`3.5527136788005009e-15`、非有限値は0で、差は浮動小数点丸めの範囲だった。

GB200 の同一入力を3回ずつ実行した中央値では、warm-up 後の device tendency が
GEMM の 20.912 ms/stage に対して GEMM_FUSED は 20.453 ms/stage で、2.20%短い。

## 2. 測定条件

- 日付: 2026-08-28（Asia/Tokyo）
- ブランチ: `feature/p511`
- base commit: `70ed8424ceb4c8622214d16855b608bd246218cb`
- 測定状態: 本レポートと同時にコミットする working tree
- 測定対象差分: `mod_advect3d_eq.f90` と p=575 入力2本
- GPU: NVIDIA GB200、189471 MiB
- 使用GPU UUID: `GPU-c2143dd7-8943-be76-38c3-7d7bd71f8c9f`
- driver: 580.173.02
- compiler: NVIDIA HPC SDK 26.3
- target: CUDA Fortran `cc100`、C++ CUDA `sm_100`
- Slurm job: なし（通常の login-node GPU 実行。`nsys` / `ncu` は未使用）
- CUDA graph: off
- profiler: なし

ビルドコマンド:

```bash
module load nvhpc
make clean
make -j4 CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100 \
  CUTLASS_HOME=/data1/rkp00015/rku00044/scale-dg-kernel-extraction/third_party/cutlass
```

非CUDA interface build も次で確認した。

```bash
make clean
make -j4
```

最終的な working executable は上記 CUDA build である。

## 3. 実装とハードウェア上の成立条件

`Ne=1`, `Nq=576`, `Np=576^3=191102976` で、既存の可変次数 GEMM を使う。

- x: `C(576, 576^2) = D1D(576,576) * flux_x(576,576^2)`
- y: 576 batch の `576 x 576 x 576` GEMM
- z: `C(576^2,576) = flux_z(576^2,576) * D1D_tr(576,576)`

`Nq=576` は 64 の9倍かつ16の36倍なので、既存の CUTLASS tile と FP64 MMA の
端数処理を必要としない。y GEMM の batch 数 `Nq*Ne=576` も CUDA grid.z の
上限65535を十分下回る。64-bit 安全化済みの host-side extent / offset を使い、
device kernel の `Np=191102976` は32-bit signed integer範囲内である。

`CUDAFORTRAN_GEMM` は3方向を cuBLAS で評価し、separable lift と assembly を
行う。`CUDAFORTRAN_GEMM_FUSED` は x/y を CUTLASS、z を custom epilogue 付き
CUTLASS で評価し、z 出力時に x/y 微分、点ごとの `Escale`、全6面の lift を
組み立てて `dqdt` を直接書く。どちらも点ごとの `q*u`, `q*v`, `q*w` と M/P
両側の数値流束を保持している。

p=575、`Ne=1` の主要 device allocation の理論値は次である。小さい演算子、
mapping、runtime allocation は含めない。

| 経路 | 主要配列 | bytes | GiB |
|---|---:|---:|---:|
| `GEMM` | `160*Np` | 30,576,476,160 | 28.48 |
| `GEMM_FUSED` | `144*Np` | 27,518,828,544 | 25.63 |
| 差 | `16*Np` | 3,057,647,616 | 2.85 |

差は GEMM_FUSED が不要とする `deriv_z` と `surface_lift` の2配列である。
`q0` と `dqdt` は owned 領域だけ、`q/u/v/w` は packed halo を含む2要素相当の
領域を確保する現行実装を前提に数えた。

**現行treeでの訂正（2026-08-28、`p1023_gap_study.md`）:** 上表はこの測定時点の
allocationを記録したものである。p=1023対応で通常GEMMも未使用`surface_lift`を
除き、z-GEMM出力を`dqdt`に置いて`deriv_z`を再利用するようになった。現行treeの
通常GEMMはGEMM_FUSEDと同じ`144*Np` = 25.63 GiBである。浮動小数点演算と定常時の
write/read回数は変わらないため、§5の測定表は当時の結果として維持する。

再現用の1ステップ入力として次を追加した。

- `input_p575_val_gemm.conf`
- `input_p575_val_gemm_fused.conf`

## 4. 数値検証

### 4.1 方法

各経路の最初の RK stage が計算した owned `dqdt(:,1:Ne)` を一時ファイルへ出力し、
全191,102,976個の FP64 値を逐次比較した。両実行に
`SCALE_DG_VARYING_COEFF=1` を指定し、点変化する速度・幾何係数・面係数を使用した。
一時 dump と比較プログラムは検証後に削除した。

### 4.2 結果

| 係数 | 比較点数 | exact 不一致 | 非有限値 | 最大絶対差 | 最大相対差 |
|---|---:|---:|---:|---:|---:|
| 点変化 | 191,102,976 | 83,136,618 | 0 | 3.5527136788005009e-15 | 4.1358014856910329e-9 |

最大相対差は値が0に近い点で生じる。最大絶対差は数 ulp の範囲であり、異なる
GEMM 縮約順による丸め差と整合する。全 owned field と全6面を含む比較なので、
定数速度や代表スカラーへの特殊化はない。

通常の benchmark 係数でも追加した両1-step入力が完走し、最終値の min/max は
両経路でそれぞれ `-9.999425159858519E-01` と
`9.999425167637535E-01` だった。

## 5. 性能

### 5.1 入力と測定方法

測定用一時入力は `/tmp/conf_perf_p575_gemm.conf` と
`/tmp/conf_perf_p575_gemm_fused.conf` とし、違いは `DqdtKernel_Type` だけである。

```fortran
NeX = 1; NeY = 1; NeZ = 1
PolyOrder = 575
dt = 1.0D-8
nstep = 50
output_interval = 50
WarmupStep = 5
UseCudaGraph = .false.
MeasureKernelTime = .true.
CublasEmulation = .false.
CutlassMmaShape = "8x8x4"
```

各経路を3回ずつ実行した。各runでは先頭5 stepを実行するが計時から除き、残り
45 step = 135 RK stages を集計した。表は3 runの中央値である。

### 5.2 結果

| 経路 | Main [s] | Main [ms/step] | device tendency [s] | device [ms/stage] |
|---|---:|---:|---:|---:|
| `CUDAFORTRAN_GEMM` | 2.92317 | 64.9594 | 2.82311 | 20.9119 |
| `CUDAFORTRAN_GEMM_FUSED` | 2.86104 | 63.5787 | 2.76111 | 20.4527 |

GEMM_FUSED は device tendency で2.20%、end-to-end Main/stepで2.13%短い。
GEMM_FUSED 内の volume GEMM 区間は中央値2.55809 s、18.9488 ms/stageだった。

各runの device tendency 合計は GEMM が 2.82237 / 2.82329 / 2.82311 s、
GEMM_FUSED が 2.76123 / 2.76111 / 2.76105 s であり、run間変動より経路差が大きい。

この測定では `ncu` を使っていないため、profiler measured FLOP/s や bandwidth は
報告しない。理論 volume work は3方向合計 `6*Nq^4 = 660,451,885,056`
FLOP/stage だが、device tendency には pointwise flux、全6面の numerical flux、
lift、assembly も含まれる。

## 6. 結論と残る課題

- p=575 は現行の generic GEMM 実装、batch上限、32-bit device index の範囲で
  ハードウェア実行可能である。
- 数値的には点変化係数を含む全 owned `dqdt` が丸め誤差範囲で一致した。
- p=575 でも最速は `CUDAFORTRAN_GEMM_FUSED` だが、優位は2.20%である。
- 本結果は p=575、`Ne=1` のもの。p=1023 では `Np=1024^3=2^30` 自体は
  32-bitに収まるが、その倍数となる extent / offset が境界を越える。既知の
  host-side境界は対処済みだが、残る32-bit device添字とメモリ容量を別途検証する必要がある。

上記p=1023の残課題は`p1023_gap_study.md`で解決し、両GEMM経路を実機検証した。

## 8. Ozaki Scheme I / II（2026-08-28 追記）

p=511 と同様に Ozaki 経路を開放し、同一 DOF（`Ne=1`）で計測。
commit `38952e4`、`nstep=20`、`WarmupStep=2`、
`OzakiSliceCount=8` / `OzakiModuliCount=14`、GB200 login ノード。

| 経路 | µs/stage | native GEMM 比 |
|---|---:|---:|
| `CUDAFORTRAN_GEMM` | **63.1 ms** | 1.00 |
| `CUDAFORTRAN_GEMM_OZAKI1` | **65.8 ms** | **1.04×** |
| `CUDAFORTRAN_GEMM_OZAKI2` | **182.4 ms** | **2.9×** |

Scheme I は native GEMM と **実質互角**（1.04×）。最速は `GEMM_FUSED`
（20.5 ms/stage）のまま。
