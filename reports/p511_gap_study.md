# p=511 GEMM / GEMM_FUSED 対応

## 1. 結論

`PolyOrder=511`（`Nq=512`）を `CUDAFORTRAN_GEMM` と
`CUDAFORTRAN_GEMM_FUSED` で実行可能にした。既存の cuBLAS / CUTLASS
カーネルは runtime `Nq` で書かれており、`Nq*Ne=512` は CUTLASS batched
GEMM の `grid.z <= 65535` 制約内にある。必要だった変更は、高次数で支配的に
なる halo の過剰確保を除くことと、p=511 を対応済みの2経路だけへ明示的に
制限することである。

点ごとに変化する `u`, `v`, `w`, `Escale`, `normal_fn`, `Fscale` を使い、
GEMM と GEMM_FUSED の全 owned `dqdt` 134,217,728 点を比較した。最大絶対差は
`3.5527136788005009e-15`、非有限値は0で、差は浮動小数点丸めの範囲だった。

GB200 の同一入力では、warm-up 後の device tendency が GEMM の
13.528 ms/stage に対して GEMM_FUSED は 13.159 ms/stage で、2.73%短い。
p=511 でも GEMM_FUSED が速いが、差は p=255 ほど大きくない。

## 2. 測定条件

- 日付: 2026-08-28（Asia/Tokyo）
- ブランチ: `feature/p511`
- base commit: `a8e0f1a31c44c44855ab70989df6cc7f04e56068`
- 測定状態: 未コミット working tree
- 測定対象ソース差分: `mod_mesh.f90`, `mod_advect3d_eq.f90`
- 上記2ファイルの `git diff` SHA-256:
  `f6e942defafcc5a380141d9f0ee64a5db711fcfe6c92a8876848b997c6a5819f`
- GPU: NVIDIA GB200、189471 MiB
- GPU UUID: `GPU-c2143dd7-8943-be76-38c3-7d7bd71f8c9f`
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

## 3. 実装

### 3.1 GEMM の形状

`Ne=1`, `Nq=512`, `Np=512^3=134217728` で、体積微分は既存の可変次数
GEMM をそのまま使う。

- x: `C(512, 512^2) = D1D(512,512) * flux_x(512,512^2)`
- y: 512 batch の `512 x 512 x 512` GEMM
- z: `C(512^2,512) = flux_z(512^2,512) * D1D_tr(512,512)`

`CUDAFORTRAN_GEMM` は3方向を cuBLAS で評価し、その後に separable lift と
assembly を1カーネルで行う。`CUDAFORTRAN_GEMM_FUSED` は x/y を CUTLASS、z
を custom CUTLASS epilogue 付きで評価し、z の出力時に x/y 微分、点ごとの
`Escale`、全6面の lift を組み立てて `dqdt` を直接書く。どちらも点ごとの
`q*u`, `q*v`, `q*w` と M/P 両側の数値流束を保持している。

### 3.2 packed halo allocation

従来は boundary element 数 `Nhalo` をそのまま `NeA=Ne+Nhalo` に加えていた。
しかし halo は完全な要素ではなく、`Np*Ne` の直後に `NhaloNode=Nfp*Nhalo`
個の面点だけを連続格納する。

p=511、`NeX=NeY=NeZ=1` では次になる。

| 項目 | 値 |
|---|---:|
| `Np` | 134,217,728 |
| `Nfp` | 262,144 |
| `Nhalo` | 6 |
| `NhaloNode` | 1,572,864 |
| 旧 `NeA` | 7 |
| 新 `NeA` | 2 |

新しい式は

```fortran
NeA = Ne + (NhaloNode + Np - 1) / Np
```

である。`q`, `q0`, `dqdt`, `u`, `v`, `w` の6配列について、host と device
それぞれの allocation は 42 GiB から 12 GiB へ減り、各側30 GiBを削減する。
halo の論理配置、`VMapP`、`halo_src_map`、RK stage ごとの `q` halo 更新は
変えていない。

### 3.3 dispatch guard と入力

p=511 は現在 `CUDAFORTRAN_GEMM` と `CUDAFORTRAN_GEMM_FUSED` のみ許可する。
再現用の1ステップ入力として次を追加した。

- `input_p511_val_gemm.conf`
- `input_p511_val_gemm_fused.conf`

LGL nodes、`D1D`、`D1D_tr`、`Lift1D(512,6)` は起動時に生成するため、巨大な
`operator_data/p511.dat` や dense `Lift_mat(512,512,512,6)` は不要である。

## 4. 数値検証

### 4.1 方法

各経路の最初の RK stage が計算した owned `dqdt(:,1:Ne)` を一時的な
unformatted stream として出力し、全134,217,728個の FP64 値を chunk 読みして
比較した。一時 dump と比較プログラムは検証後に削除した。

実行コマンド:

```bash
env CUDA_VISIBLE_DEVICES=0 SCALE_DG_DUMP_DQDT=/tmp/p511_gemm_dqdt.bin \
  ./scale-dg_extraction input_p511_val_gemm.conf
env CUDA_VISIBLE_DEVICES=0 SCALE_DG_DUMP_DQDT=/tmp/p511_gemm_fused_dqdt.bin \
  ./scale-dg_extraction input_p511_val_gemm_fused.conf
```

点変化係数では両コマンドに `SCALE_DG_VARYING_COEFF=1` を追加した。

### 4.2 結果

| 係数 | 比較点数 | exact 不一致 | 非有限値 | 最大絶対差 | 最大相対差 |
|---|---:|---:|---:|---:|---:|
| benchmark 定数 | 134,217,728 | 36,713,396 | 0 | 8.8817841970012523e-16 | 6.7309659832256207e-11 |
| 点変化 | 134,217,728 | 58,379,785 | 0 | 3.5527136788005009e-15 | 2.3544598113231188e-9 |

最大相対差は値が0に近い点で生じる。最大絶対差はどちらも数 ulp の範囲であり、
GEMM 実装の異なる縮約順による丸め差と整合する。点変化ケースでも全6面を含む
完全な `dqdt` が一致しており、定数速度や代表スカラーへの特殊化はない。

packed halo の回帰として p=7 `CUDAFORTRAN_GEMM` と p=255
`CUDAFORTRAN_GEMM_FUSED` も1ステップ実行し、いずれも成功した。

## 5. 性能

### 5.1 入力

測定入力名は他次数の作業用設定に合わせて `/tmp/conf_perf_p511.conf` と
`/tmp/conf_perf_p511_fused.conf` とした。両者の差は `DqdtKernel_Type` だけで、
共通条件は次である。

```fortran
NeX = 1; NeY = 1; NeZ = 1
PolyOrder = 511
dt = 1.0D-8
nstep = 20
output_interval = 20
WarmupStep = 2
UseCudaGraph = .false.
MeasureKernelTime = .true.
CublasEmulation = .false.
CutlassMmaShape = "8x8x4"
```

実行コマンド:

```bash
env CUDA_VISIBLE_DEVICES=0 ./scale-dg_extraction /tmp/conf_perf_p511.conf
env CUDA_VISIBLE_DEVICES=0 ./scale-dg_extraction /tmp/conf_perf_p511_fused.conf
```

### 5.2 結果

先頭2ステップを実行するが計時から除き、残り18ステップ = 54 RK stages を
集計した。

| 経路 | Main [s] | Main [ms/step] | device tendency [s] | device [ms/stage] |
|---|---:|---:|---:|---:|
| `CUDAFORTRAN_GEMM` | 0.752467 | 41.8037 | 0.730525 | 13.5282 |
| `CUDAFORTRAN_GEMM_FUSED` | 0.732586 | 40.6992 | 0.710564 | 13.1586 |

GEMM_FUSED は device tendency で2.73%、end-to-end Main/step で2.64%短い。
GEMM_FUSED 内の volume GEMM 区間は合計0.652422 s、12.0819 ms/stage だった。

この測定では `ncu` を使っていないため、FLOP/s、帯域、profiler measured
operation count は報告しない。理論 volume work は3方向合計
`6*Nq^4 = 412,316,860,416` FLOP/stage だが、device tendency 時間には
pointwise volume flux、全6面の numerical flux、lift、assembly も含まれるため、
この理論値だけから profiler 実測 FLOP/s を装うことはしない。

## 6. 残る課題

- 性能差は2.7%なので、安定した最終値には `ncu` で x/y/z GEMM と epilogue の
  内訳を採る必要がある。profiler を使う場合は Slurm、frozen executable、
  `timeout`、job 内の `module load nvhpc` を使う。
- p=511 の `Ne>1` は整数添字と batch 上限の範囲では実行可能だが、今回は
  intended case の `Ne=1` だけを実測した。追加要素はメモリ量が大きいため、
  smoke test は空きGPUメモリを確認して行う。
- `surface_lift` の未使用 allocation は本変更で GEMM_FUSED と GEMM_CUTE から除去した。
  GEMM_CUTE も `separable_lift_assembly_kernel` が直接 assembly するため、不要な
  `lift_out` interface も併せて削除した。p=511 `Ne=1` では 1配列あたり 1 GiB
  （一般には `8*Np*Ne` bytes）の device memory を予約していたが、両経路とも
  lift と assembly を device kernel 内で完結するため定常 kernel のデータ経路は変わらない。
  p=7 / p=511 の smoke test と
  CUDA / 非CUDA clean build を再確認した。変更の性能・数値比較は
  `reports/index64_boundary_validation.md` に記録した。

**現行treeでの追記（2026-08-28、`p1023_gap_study.md`）:** 通常GEMMもliftと
assemblyを既に融合済みだったため、未使用`surface_lift`を除いた。さらにz-GEMM
出力を`dqdt`へ直接置き、assemblyが同じ場所を読んで上書きすることで`deriv_z`も
除いた。p=511では各1 GiB、計2 GiBのdevice allocation削減である。演算順と定常時の
write/read回数は変わらず、p=1023の全点比較とp=511点変化smoke testで確認した。
