# p=767 GEMM / GEMM_FUSED 対応

## 1. 結論

`PolyOrder=767`（`Nq=768`）を `CUDAFORTRAN_GEMM` と
`CUDAFORTRAN_GEMM_FUSED` で実行可能にした。既存の cuBLAS / CUTLASS
カーネルは runtime `Nq` で実装済みであり、必要なコード変更は p=767 を
対応済みの2経路だけに制限する dispatch guard の追加である。

点ごとに変化する `u`, `v`, `w`, `Escale`, `normal_fn`, `Fscale` を使い、
両経路の全 owned `dqdt` 452,984,832 点を比較した。最大絶対差は
`3.5527136788005009e-15`、非有限値は0で、差は浮動小数点丸めの範囲だった。

GB200 の同一入力を3回ずつ実行した中央値では、warm-up 後の device tendency が
GEMM の 62.817 ms/stage に対して GEMM_FUSED は 60.362 ms/stageで、3.91%短い。

## 2. 測定条件

- 日付: 2026-08-28（Asia/Tokyo）
- ブランチ: `feature/p511`
- base commit: `bc5aac4`（`Add p=575 GEMM paths and report`）
- 測定状態: 本レポートと同時にコミットする working tree
- 測定対象差分: `mod_advect3d_eq.f90` と p=767 入力2本
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

`Ne=1`, `Nq=768`, `Np=768^3=452984832` で、既存の可変次数 GEMM を使う。

- x: `C(768, 768^2) = D1D(768,768) * flux_x(768,768^2)`
- y: 768 batch の `768 x 768 x 768` GEMM
- z: `C(768^2,768) = flux_z(768^2,768) * D1D_tr(768,768)`

`Nq=768` は64の12倍かつ16の48倍なので、既存の CUTLASS tile と FP64 MMA の
端数処理を必要としない。y GEMM の batch 数 `Nq*Ne=768` も CUDA grid.z の
上限65535を十分下回る。`Np=452984832`、packed haloを含む主場の extent
`2*Np=905969664`、`Escale` の最大方向offset `2*Np=905969664` はいずれも
32-bit signed integer範囲内である。

`CUDAFORTRAN_GEMM` は3方向を cuBLAS で評価し、separable lift と assembly を
行う。`CUDAFORTRAN_GEMM_FUSED` は x/y を CUTLASS、z を custom epilogue 付き
CUTLASS で評価し、z 出力時に x/y 微分、点ごとの `Escale`、全6面の lift を
組み立てて `dqdt` を直接書く。どちらも点ごとの `q*u`, `q*v`, `q*w` と M/P
両側の数値流束を保持している。

p=767、`Ne=1` の主要 device allocation の理論値は次である。小さい演算子、
mapping、runtime allocation は含めない。

| 経路 | 主要配列 | bytes | GiB |
|---|---:|---:|---:|
| `GEMM` | `160*Np` | 72,477,573,120 | 67.50 |
| `GEMM_FUSED` | `144*Np` | 65,229,815,808 | 60.75 |
| 差 | `16*Np` | 7,247,757,312 | 6.75 |

差は GEMM_FUSED が不要とする `deriv_z` と `surface_lift` の2配列である。
`q0` と `dqdt` は owned 領域だけ、`q/u/v/w` は packed halo を含む2要素相当の
領域を確保する現行実装を前提に数えた。

**現行treeでの訂正（2026-08-28、`p1023_gap_study.md`）:** 上表はこの測定時点の
allocationを記録したものである。p=1023対応で通常GEMMも未使用`surface_lift`を
除き、z-GEMM出力を`dqdt`に置いて`deriv_z`を再利用するようになった。現行treeの
通常GEMMはGEMM_FUSEDと同じ`144*Np` = 60.75 GiBである。浮動小数点演算と定常時の
write/read回数は変わらないため、§5の測定表は当時の結果として維持する。

再現用の1ステップ入力として次を追加した。

- `input_p767_val_gemm.conf`
- `input_p767_val_gemm_fused.conf`

## 4. 数値検証

### 4.1 方法

各経路の最初の RK stage が計算した owned `dqdt(:,1:Ne)` を一時ファイルへ出力し、
全452,984,832個の FP64 値をストリーム比較した。両実行に
`SCALE_DG_VARYING_COEFF=1` を指定し、点変化する速度・幾何係数・面係数を使用した。
各dumpは452,984,832行、約11 GBであることを確認し、比較後に削除した。

### 4.2 結果

| 係数 | 比較点数 | exact 不一致 | 非有限値 | 最大絶対差 | 最大相対差 |
|---|---:|---:|---:|---:|---:|
| 点変化 | 452,984,832 | 197,424,052 | 0 | 3.5527136788005009e-15 | 1.2216460347937831e-9 |

最大相対差は値が0に近い点で生じる。最大絶対差は数 ulp の範囲であり、異なる
GEMM 縮約順による丸め差と整合する。全 owned field と全6面を含む比較なので、
定数速度や代表スカラーへの特殊化はない。

通常の benchmark 係数でも追加した両1-step入力が完走し、最終値の min/max は
両経路でそれぞれ `-9.999676850160175E-01` と
`9.999676855992745E-01` だった。

## 5. 性能

### 5.1 入力と測定方法

測定用一時入力は `/tmp/conf_perf_p767_gemm.conf` と
`/tmp/conf_perf_p767_gemm_fused.conf` とし、違いは `DqdtKernel_Type` だけである。

```fortran
NeX = 1; NeY = 1; NeZ = 1
PolyOrder = 767
dt = 1.0D-8
nstep = 30
output_interval = 30
WarmupStep = 5
UseCudaGraph = .false.
MeasureKernelTime = .true.
CublasEmulation = .false.
CutlassMmaShape = "8x8x4"
```

各経路を3回ずつ実行した。各runでは先頭5 stepを実行するが計時から除き、残り
25 step = 75 RK stagesを集計した。表は3 runの中央値である。

### 5.2 結果

| 経路 | Main [s] | Main [ms/step] | device tendency [s] | device [ms/stage] |
|---|---:|---:|---:|---:|
| `CUDAFORTRAN_GEMM` | 4.80593 | 192.237 | 4.71126 | 62.8168 |
| `CUDAFORTRAN_GEMM_FUSED` | 4.62421 | 184.968 | 4.52715 | 60.3620 |

GEMM_FUSED は device tendency で3.91%、end-to-end Main/stepで3.78%短い。
GEMM_FUSED 内の volume GEMM 区間は中央値4.25913 s、56.7884 ms/stageだった。

各runの device tendency 合計は GEMM が 4.71132 / 4.71126 / 4.71099 s、
GEMM_FUSED が 4.52647 / 4.52715 / 4.52822 sであり、run間変動より経路差が大きい。

この測定では `ncu` を使っていないため、profiler measured FLOP/s や bandwidth は
報告しない。理論 volume work は3方向合計 `6*Nq^4 = 2,087,354,105,856`
FLOP/stageだが、device tendencyには pointwise flux、全6面の numerical flux、
lift、assemblyも含まれる。

## 6. 結論と残る課題

- p=767 は現行の generic GEMM 実装、batch上限、32-bit device indexの範囲で
  ハードウェア実行可能である。
- 数値的には点変化係数を含む全 owned `dqdt` が丸め誤差範囲で一致した。
- p=767 でも最速は `CUDAFORTRAN_GEMM_FUSED` で、優位は3.91%である。
- 本結果は p=767、`Ne=1` のもの。p=1023では `Np=2^30` 自体は32-bitに収まるが、
  その倍数となる extent / offset が境界を越える。既知のhost-side境界は対処済みだが、
  残る32-bit device添字とメモリ容量を別途検証する必要がある。

上記p=1023の残課題は`p1023_gap_study.md`で解決し、両GEMM経路を実機検証した。
