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

## 8. Ozaki Scheme I / II（2026-08-28 追記）

`mod_advect3d_eq.f90` の p=511 ゲートに `CUDAFORTRAN_GEMM_OZAKI1/2` を追加し、
同一 DOF（`Ne=1`）で `CUDAFORTRAN_GEMM` を基準に計測。commit `38952e4`、
`nstep=20`、`WarmupStep=2`（§6 と同条件）、
`OzakiSliceCount=8` / `OzakiModuliCount=14`、GB200 login ノード。

| 経路 | µs/stage | native GEMM 比 |
|---|---:|---:|
| `CUDAFORTRAN_GEMM` | **40.0 ms** | 1.00 |
| `CUDAFORTRAN_GEMM_OZAKI1` | **45.7 ms** | **1.14×** |
| `CUDAFORTRAN_GEMM_OZAKI2` | **124.7 ms** | **3.1×** |

[`ozaki2_survey_2504.08009.md`](ozaki2_survey_2504.08009.md) の成立条件
（演算強度 ≳ 3.82 FLOP/byte、p ≳ 500–650）に p=511 が近づき、
**Scheme I が native volume GEMM に 1.14 倍**まで縮む。最速は `GEMM_FUSED`
（13.2 ms/stage）のまま。

> **（訂正 2026-08-29）** 上表の native **40.0 ms** は 1 step の device 合計で、
> RK 3 stage で割っていない。1 stage は約 13.3 ms で、§5 の GEMM 13.528 ms/stage
> と整合する。Ozaki 比はそのまま読める。

> **（訂正 2026-09-03）** 2 点直す。表そのものは当時の測定として残す。
>
> 1. **commit は `38952e4` ではなく `dd3c4a7`。** p≥511 の Ozaki ゲートを
>    開けたのは `dd3c4a7` で、本節の本文を追加したのもその commit である
>    （`38952e4` はその親で、そこでは p=511 の Ozaki は `error stop` する）。
> 2. **上表は `EmulationMantissaControl` が DYNAMIC（ADP）だった時期の値である。**
>    既定は `fd091fc`（2026-08-29）で FIXED 55 bit に変わり、同じ制御が
>    Ozaki-I の残差 early-exit と Ozaki-II の追加 A パックを決めている。
>    現行ツリー `44b02a6` で ADP に戻すと **`OZAKI1` 1.140× / `OZAKI2` 3.114×**
>    と上表を再現し、既定の FIXED では **16.6× / 2.37×** になって順位が逆転する。
>    さらに **ADP の `OZAKI1` は owned `dqdt` が合わない**（p=7 で 3.04e-01、
>    実効スライス 1.33 枚）ので、**上表の 1.14× は DG の要求精度を
>    満たさない設定の速度**である。精度を満たすのは `OZAKI1` FIXED だけで、
>    p=511 では native の 16.6× になる。機構と再現の全記録は
>    [`gemm_assignment_and_carrier.md`](gemm_assignment_and_carrier.md) §9.8。

## 9. cuBLAS FP64 emulation（2026-08-28 追記）

`CublasEmulation = .true.` で `CUDAFORTRAN_GEMM` の volume 3 方向を cuBLAS
fixed-point emulation（EAGER）に置き換えた。commit `a1cdb57`、§5 と同条件
（`nstep=20`、`WarmupStep=2`）、GB200 login ノード、3 run 中央値。

| 経路 | device [ms/stage] | native GEMM 比 |
|---|---:|---:|
| `CUDAFORTRAN_GEMM`（native） | 13.53 | 1.00 |
| `CUDAFORTRAN_GEMM` + emulation | **20.23** | **1.50×** |

p=255 の 8.8× より大幅に縮むが、依然 native より遅い。入力は
`input_p511_val_gemm_emu.conf`。詳細は
[`cublas_emulation_survey.md`](cublas_emulation_survey.md) §4。

## 10. 2026-08-29 の経路横断について

p=7…255 は同一実行ファイルで再測定したが、**p=511 は再実行していない**。
[`README.md`](README.md) まとめ表の p=511 行は本レポート §5 の
`GEMM` 13.528 / `GEMM_FUSED` 13.159 ms/stage のまま。

> **（訂正 2026-09-01）** 上の README 行は §11 で更新した。§5 の 13.159 ms は
> 当時の実行ファイルの値として残す。現行 tree の `GEMM_FUSED` は 12.48 ms/stage
> 前後で、この差は p=511 専用の変更ではなく、既に入っていた z 融合エピローグ
> （`Escale` 前送り、`deriv_x` の y 畳み込み、16 B epilogue）による。
> `GEMM` 13.528 は再測定していない。

## 11. `GEMM_FUSED` 探索（2026-08-30〜09-01）

対象は `CUDAFORTRAN_GEMM_FUSED`、`namelists/perf_p511_gemm_fused.conf`
（`Ne=1`、p=511、`nstep=20`、`WarmupStep=2`、graph off、`8x8x4`）。
親 commit `f7788bb`。この節で採択したソース変更は無い。

### 11.1 ベースライン

凍結実行ファイル `scale-dg_extraction.p511gf`（2026-08-30 21:10、login GPU、
3-run）。device fused 0.673985 / 0.674076 / 0.674103 s（54 stage = 18 step × 3）。
中央値 **0.674076 s → 12.483 ms/stage**、Main **38.68 ms/step**、
volume GEMM only 11.418 ms/stage。§5 の 13.159 ms より約 5% 短い。

占有 GPU の交互 A/B でも同じ桁で、ノード差は 0.4% 級（job `73618` A 中央値
12.495 ms、job `74696` A 12.449 ms）。以降の採否は各ジョブ内の対 A 比。

3 本の volume GEMM の演算下限は `2*Nq^4 / 40.1e12` = **3.427 ms/本**。

### 11.2 律速（機構。採否は実時間）

**nsys job `71126`**（c178、nstep=4、凍結 exe、中央値）:

| カーネル | 時間 |
|---|---:|
| z `GemmBatchedDqdtAssembly` 64×32 | 3.867 ms（下限の 88.7%） |
| x CUTLASS `Gemm` 64×128 | 3.681 ms（93.2%） |
| y `GemmBatchedScaleAdd` | 3.624 ms（94.3%） |
| `volume_flux_kernel` | 1.036 ms |
| `elembnd_flux_kernel` | 0.099 ms（side stream で隠れている） |
| RK 更新（tendency 外） | ~0.60–0.63 ms |

**ncu job `73614`**（c182、`--set full`、`-s 30 -c 5`。x/y は skip の関係で未採取）:

- **z**: SM 91.57%、DRAM 7.54%、占有 12.42%、254 reg、Shared(FP64/Tensor) pipe 91.6%。Duration 7.2 ms は ncu 固定クロック。
- **flux**: DRAM **91.24%**、SM 44%、24 reg、占有 80.5%、~1.03 ms。
- **elembnd**: DRAM 73%、76 µs。

z は演算パイプ、flux は DRAM、elembnd は既に隠れている。

### 11.3 不採用（実時間 A/B、すべて戻した）

| 候補 | 仮説 | 測定 | 結果 |
|---|---|---|---|
| Nq>64 の x を cuBLAS に | p=127 では K が深いと cuBLAS が追いつく | login 交互 3 回。A 0.674 vs B 0.678–0.679 | **+0.6% 級。不採用** |
| flux_x 先行 + flux_yz を x GEMM 裏（Nq≥512） | nsys 上 flux 1.04 ms が直列の最大塊。p=255/63 では x の DRAM 争いで負けたが、ここは A が 1 GiB | login 交互。A すべて ~0.674、B 0.692–0.708。volume GEMM only 0.616→0.663 | **device +2.6%。不採用**（x が DRAM 争いで伸びる。p=255/63 と同じ機構） |
| z タイル 64×64（Nq≥512） | p=127 では L2 が `flux_z` を掴むので無効。p=511 の A は L2 に載らない | job **`73618`**（c179、12 交互）。A 64×32 中央値 0.674731 s（12.495 ms、0.674598–0.675095）。B 64×64 0.677730 s（12.551 ms、0.677588–0.678013） | **+0.44%、レンジ非重複。不採用**。ncu の z は DRAM 7.5% でパイプ 91.6% が律速。A の再読仮説は外れ |
| x GEMM 4-stage（Nq≥512） | K=512 は TileK=16 で 32 反復。stage 3→4 は占有をほぼ払わず A/B 往復を隠せる | job **`74696`**（c399、12 交互）。A 3-stage 中央値 0.672259 s（12.449 ms、0.672168–0.672363）。B 4-stage 0.673554 s（12.473 ms、0.673472–0.673700） | **+0.193%、レンジ非重複。不採用**。p=127 の x 4-stage 負けと同じ向き |

### 11.4 天井だけのアブレーション（実装しない）

z エピローグから lift を消す（`-DDG_Z_SKIP_LIFT=1`、数値不正）。
job **`73627`**（c179、12 交互）:

| | device 中央値 | ms/stage |
|---|---:|---:|
| A lift あり | 0.671814 s | 12.441 |
| B lift なし | 0.664351 s | 12.303 |

device **−1.11%**（138 µs/stage）、gemm-only −1.21%、レンジ非重複。
lift は数値契約上必須なので採用しない。ifdef は戻した。

### 11.5 残る天井と終了

契約内でまだ取れていないもの:

| 天井 | 大きさ | 取り方 | 判定 |
|---|---:|---|---|
| x/y を演算下限へ | 各 ~0.25 ms | ライブラリ mainloop。cuBLAS x と 4-stage は上で負け | 測った。残は CUTLASS 本体 |
| z mainloop + 必須エピローグ | 下限との差 0.44 ms のうち lift 以外 ~0.30 ms | `Escale` / 面 lift / `dqdt` 書きは契約。タイル 64×64 は負け | 契約内ノブは尽きた |
| lift 全消し | **138 µs（1.11%）** | 面 lift を捨てる | **範囲外** |
| flux を DRAM 100% へ | ~90 µs（0.72%） | ベクトル化・C++・CTA 形。同一カーネルを p=63 `GEMM_FUSED` が DRAM 83%→100% で全部損か引き分け（`p63_gap_study.md` §41）。ここは既に DRAM 91% なので残はさらに狭い | 同じ機構。再実装しない |
| flux を GEMM に畳む / yz を x 裏 | flux 1.04 ms | 方向分割重ねは +2.6% | 不採用済み |
| RK 更新 | ~0.6 ms/stage | `dqdt` を実体化しない | **範囲外**（配列インタフェース） |
| `q,u,v,w` パック | flux のセクタ | 呼び出し側レイアウト | **範囲外** |

CUTLASS タイル掃引を p=511 でやり直していない理由は、p=127/255 の掃引が全滅で、
ここでの x/y が既に下限の 93–94% にいること。`GEMM_CUTE` のタイルは触っていない。

**最速は `CUDAFORTRAN_GEMM_FUSED` のまま。** 現行 tree の公開値は
**12.48 ms/stage** / Main **38.68 ms/step**（§11.1）。ソース差分は無い。

## 12. y GEMM のパイプライン段数（2026-09-01）

対象は `CUDAFORTRAN_GEMM_FUSED`、`namelists/perf_p511_gemm_fused.conf`
（`Ne=1`、p=511、`nstep=20`、`WarmupStep=2`、graph off、`8x8x4`）。
親 commit は `feature/cuda` へ rebase 後の `acdbd8a`。GPU は GB200、
compiler は NVIDIA HPC SDK 26.3、`cc100` / `sm_100`。

### 12.1 rebase 後のベースライン

`feature/cuda` が持ち込んだ変更のうち p=511 に届くのは、y GEMM の
`RepadEpilogue<..., 8>`（`run_gemm_batched_nn_capped`）だけである。p=7 で
−4.1%、p=15 で −0.6% だったこのパッドは、**Nq=512 では中立**だった:
login 3-run で device fused 0.673959 / 0.674108 / 0.674072 s、中央値
**0.674072 s → 12.483 ms/stage**、Main 38.68 ms/step。§11.1 の
0.674076 s と 4 µs しか違わない。§11 の分母はそのまま使える。

### 12.2 採用: 融合 y GEMM を 4 段から 3 段へ

`VolumeGemmSet::GemmYScale` の stage 数を 4 → 3 にする（`GemmYScaleShallow`
を新設し、`run_volume_gemm_xy` の Nq > 64 経路だけがこれを取る）。同時に
未融合対照の `GemmY` も 3 段にした。

p=511 の採否計測（占有 GPU、10 交互、job **`74863`**、c179）:

| | device fused 中央値 | ms/stage | Main/step | レンジ |
|---|---:|---:|---:|---|
| A 4-stage | 0.671772 s | 12.440 | 38.554 ms | 0.671724–0.672022 |
| B 3-stage | **0.670343 s** | **12.414** | **38.475 ms** | 0.670202–0.670491 |

**−0.213%、レンジ非重複。** 別ノードの先行 A/B も同じ向きで、job `74818`
（c182、12 交互）−0.230%、job `74825`（c185、10 交互）−0.219%。

### 12.3 なぜ効くか（ncu job `74824`）

同一ジョブ内で A/B を並べた。y GEMM (`GemmBatchedScaleAdd`):

| | shared/CTA | 占有率 | occ 制限 (smem/reg) | 命令数 | SM 対ピーク | ncu duration |
|---|---:|---:|---:|---:|---:|---:|
| A 4-stage | 65,536 B | 18.28% | 3 / 3 | 1,163,001,856 | 94.94% | 6.8364 ms |
| B 3-stage | 49,152 B | 18.28% | 3 / 3 | 1,153,695,744 | 95.49% | 6.7948 ms |

**占有率は動いていない**（レジスタ 168 で 3 CTA が上限、shared を 16 KB
返しても 4 CTA にはならない）。動いたのは**命令数 −0.80%** と SM スループット
+0.55 ポイントだけである。つまりこの GEMM は**レイテンシ律速ではなく発行律速**
であり、4 段目は長いプロローグの代金を払うだけで何も隠していない。z
(`GemmBatchedDqdtAssembly`) は命令数まで完全に同一で、副作用は無い。

同じ機構が §11.3 の「x GEMM 4-stage が +0.193%」を説明する。3 段の y の上に
x 4-stage を重ねても job `74825` で **−0.014%**（差が無い）で、x は 3 段が最善
のままである。**2 段は CUTLASS 2.x の FP64 TensorOp では表現できない**
（`DefaultMmaCore` の部分特殊化が曖昧になりコンパイルできない）。3 段が下限。

### 12.4 次数依存（横展開）

利得は **K/TileK = Nq/16 回のループに固定のプロローグ費用を薄める**形なので、
Nq が上がるほど小さくなる。すべて占有 GPU の交互 A/B。

`CUDAFORTRAN_GEMM_FUSED`（`GemmYScale`）:

| Nq | p | job / ノード | 3-stage の効果 |
|---:|---:|---|---:|
| 64 | 63 | `74843` c190 | **+0.446%（回帰）** |
| 128 | 127 | `74833` c159 | **−0.762%** |
| 256 | 255 | `74843` c190 | **−0.603%** |
| 512 | 511 | `74863` c179 | **−0.213%** |
| 768 | 767 | `74847` c159 | **−0.104%** |
| 1024 | 1023 | `74847` c159 | **−0.061%** |

`CUDAFORTRAN_GEMM_CUTE`（`GemmY`、融合しない対照）:

| Nq | p | job / ノード | 3-stage の効果 |
|---:|---:|---|---:|
| 64 | 63 | `74844` c159 / `74863` c179 | **−0.930% / −0.781%** |
| 128 | 127 | `74844` c159 | **−0.598%** |
| 256 | 255 | `74848` c182 | **−0.271%** |

**Nq=64 だけが融合側で符号が反転する**。融合 y のエピローグは 2 ソース
（`deriv_x` を global から読む）で、最小の batch 形ではその読みを隠すのに
4 段目が要る。同じ Nq=64 でもエピローグが 1 ソースの `GemmY` は 3 段が勝つ。
そこで `GemmYScaleShallow` は **Nq > 64 の経路だけ**に入れ、Nq=64 の
`launch_volume_gemm_y_scaleadd` は 4 段のままにした。最終形の p=63
`GEMM_FUSED` は job `74863` で **+0.027%（レンジ重複、差が無い）**、
p=63 `GEMM_CUTE` は **−0.781%** である。

### 12.5 数値検証

`SCALE_DG_VARYING_COEFF=1` で全 owned `dqdt` をダンプして比較した。

| 比較 | 点数 | exact 不一致 | 非有限 | 最大絶対差 | 最大相対差 |
|---|---:|---:|---:|---:|---:|
| p=511 `GEMM` vs `GEMM_FUSED`(3段) | 134,217,728 | 43,547,223 | 0 | 1.7763568394002505e-15 | 2.3544598168666e-09 |
| p=511 `GEMM_FUSED` 4段 vs 3段 | 134,217,728 | 0 | 0 | 0（ビット一致） | 0 |
| p=255 `GEMM` vs `GEMM_CUTE`(3段) | 16,777,216 | 0 | 0 | 0（ビット一致） | 0 |
| p=255 `GEMM` vs `GEMM_FUSED`(3段) | 16,777,216 | 5,401,109 | 0 | 1.7763568394002505e-15 | 2.4910925890447103e-10 |

段数は K ループの縮約順を変えないので、4 段版とビット一致するのは想定どおり
である。`GEMM` 参照との差は §4 の 3.55e-15 より小さい。

### 12.6 不採用（すべて実測して戻した）

**CUTLASS タイル掃引**。§11.5 が「p=511 では未実施」と書いていたものを埋めた。
login 3-run（`FUSED volume GEMM only` はレンジ 0.02% と非常に安定。混雑した
run が混じるので最小値で比較した）:

| 変種 | gemm-only vs A |
|---|---:|
| z warp `32x16`（CTA 4 ワープ） | +1.25% |
| z タイル `128x32` | +0.74% |
| z stage 5 | +0.97% |
| z stage 3 | +0.03%（占有 GPU の job `74818` では **+0.144%**、不採用） |
| x タイル `128x128` | +1.41% |
| x タイル `64x256` | +1.64% |
| y タイル `128x64` | +4.08% |

**z のラスタライズ順（N 最内）**。z assembly は `tiles_m = Nq²/64 = 4096`、
`tiles_n = Nq/32 = 16` で、CUDA grid は M が最内である。「N タイルごとに
`flux_z`（1 GiB）を読み直しており L2 に載らない」という仮説で、両軸を入れ替えた
swizzle を書いた。結果は **+0.49%** で、ncu job **`74739`** が仮説を否定した:

| | z dramR | L2 hit | DRAM 対ピーク | SM 対ピーク | ncu duration |
|---|---:|---:|---:|---:|---:|
| A M 最内 | 3.242 GB | 78.75% | 7.49% | 91.49% | 7.2359 ms |
| B N 最内 | 10.783 GB | 48.47% | 20.63% | 91.40% | 7.2428 ms |

M 最内でも L2 が 16 回の読み直しのうち 13 回分を吸っており、A の DRAM は
既に 3.2 GB しかない。N 最内は逆に L2 ヒットを 79% → 48% に落として DRAM を
3.3 倍にする。**それでいて時間は +0.09% しか動かない** —— z は演算律速で、
DRAM に 3.3 倍の余裕がある。この事実は §11.3 の「z タイル 64×64 が負けた」
理由も裏づける。

**flux のベクトル化**（1 スレッドが連続 2 点 / 4 点）。flux + その他の合計が
1.0628 → **1.4396**（+35%）→ **2.1697 ms/stage**（+104%）。1 点 1 スレッドの
コアレス形が最適で、`p63_gap_study.md` §41 と同じ結論である。

**flux を volume GEMM の裏に隠す + L2 evict-first ヒット**。§11.3 の
「flux_yz を x GEMM 裏」が負けた理由を「flux が L2 を掃いて GEMM の working
set を追い出すから」と仮説し、`__ldcs` / `__stcs` で flux の行を evict-first
にする C++ カーネルを書いて 6 通りを測った（login 3-run、`SCALE_DG_FLUX_STREAM`
と `SCALE_DG_FLUX_OVERLAP`）:

| stream ヒント | 裏に回す成分 | device ms/stage | vs A | volume GEMM ms/stage | flux ほか ms/stage |
|---:|---|---:|---:|---:|---:|
| 0 | なし (A) | 12.483 | +0.00% | 11.421 | 1.0621 |
| 1 | なし | 12.701 | +1.74% | 11.419 | 1.2818 |
| 0 | z | 12.976 | +3.95% | 12.091 | 0.8844 |
| 1 | z | 13.005 | +4.18% | 12.092 | 0.9128 |
| 0 | y,z | 12.952 | +3.76% | 12.425 | 0.5271 |
| 1 | y,z | 12.978 | +3.96% | 12.438 | 0.5389 |

ヒントは**単独でも損**（flux だけで +21%。`.cs` は 90% 出ているストリームに
とっては足枷でしかない）で、**重ねたときの赤字も直さない**。裏に回して隠れた
時間に対し volume GEMM が払う代金は z だけで 0.178 : 0.670 = **1 : 3.8**、
y+z で 0.535 : 1.004 = **1 : 1.9** である。L2 の入れ替えでは説明しきれない
（ヒントで L2 を汚さなくしても同じだけ遅い）。**この一族は p=511 では閉じた。**

### 12.7 現在の内訳と残る天井

3 本の volume GEMM の演算下限は `2*Nq^4 / 40.1e12` = 3.427 ms/本。
ncu job `74739` / `74824` の対ピーク:

| ブロック | 時間 | 律速 |
|---|---:|---|
| x `GemmXScale` | 3.681 ms | SM **95.06%** |
| y `GemmBatchedScaleAdd`（3 段） | 3.60 ms | SM **95.49%** |
| z `GemmBatchedDqdtAssembly` | 3.867 ms | SM **91.52%**、DRAM 7.5% |
| `volume_flux_kernel` | 1.036 ms | DRAM **90.25%** |
| `elembnd_flux_kernel` | 0.077 ms | side stream で隠れている |

x と y は SM スループット 95% にいる。z の 3.5 ポイントの不足は §11.4 で
測った lift そのもの（138 µs = 1.11%）とほぼ一致し、これは数値契約上必須。
flux は DRAM 90% で、ベクトル化でも重ねでも縮まない。**契約内の候補は
これで尽きた。**

**最速は `CUDAFORTRAN_GEMM_FUSED` のまま。公開値は 12.414 ms/stage /
Main 38.475 ms/step**（job `74863`、c179、10 交互中央値）。

## 追記（2026-09-01）: p=575 側から入った `Nq >= 512` の分岐

[`p575_gap_study.md`](p575_gap_study.md) §11 が volume GEMM に `Nq >= 512` の
分岐を 2 つ足したので、本次数にもそのまま載る。
本レポート §12 の公開値 12.414 ms/stage は **batched x が入る前**の値である
（§12 が 4 段→3 段にしたのは融合 y だけ）。

- **§11.2**: 融合 x を `64x128` の 1 本から、y と同じ `64x64` batched へ。
- **§11.13**: その x も `GemmYScaleShallow` を共有して 3 段パイプラインにする
  （`GEMM_CUTE` の x / y も同じ mainloop）。

両方入った実行ファイルと、batched x だけ入れて x/y は 4 段のままの実行ファイルを
占有 GPU 上で 8 回交互に測った（`p575_gap_study.md` §11.16、job `74975`、
`namelists/perf_p511_gemm_fused.conf` を `nstep`/`WarmupStep` だけ変えたもの）:
**12412.9 → 12355.0 µs/stage（−0.466%）**。この A/B が測っているのは
**batched x の上で x/y を 4 段から 3 段にする効果**であって、
`p511_gap_study.md` §12 が融合 y 単独で測った値とは分母も対象も違う。
数値は p=575 で 191,102,976 点ビット一致（§11.13）。

## 13. `Nq>64` の融合パッケージは p=511 / 767 / 1023 で既に有効（2026-09-01、測定不要）

コミット: `c386c85`（本節を追加したコミットの親）。コードを読んで確定した事実で、
A/B は存在しない。

`p127_gap_study.md` §13（**p=127 −2.8%、p=255 −1.6%、p=63 はビット一致で不変**）を
「z assembly を 4 本の CUDA ストリームに分けて流す」ものだと読むと横展開先が
残っているように見えるが、**§13 が動かしたのは CUDA ストリームではなく
z エピローグが読むデータの本数**である。表題の「4 本目のストリーム」は
z が読む 5 本のオペランド（`flux_z` / `deriv_x` / `deriv_y` / `Escale_z` の読みと
`dqdt` の書き）のうち `deriv_x` を指し、それを y GEMM のエピローグ
（`cutlass_y_gemm_scaleadd.h` の `deriv_xy = Escale_y*acc + deriv_x`）へ移した。

このパッケージのゲートは Nq だけを見ており、p=511 / 767 / 1023（Nq = 512 / 768 /
1024）はすべて通っている:

| 要素 | ゲート | 場所 | Nq≥512 で |
|---|---|---|---|
| `Escale` を x/y エピローグへ、`deriv_x` を y に畳む（`xy_weighted`） | `fuse_epilogue .and. Nq >= 64` | `mod_cuda_dg_kernels.cuf:2019` | **on** |
| lift の面ペア 16 バイトロード（`pair_nq2`） | `fuse_epilogue .and. (Nq >= 64 .or. Nq == 8)` | 同 `:2023` | **on** |
| z エピローグ 16 バイトアクセス（`GemmZWide`） | `kWide = kWeighted`、`Nq == 64` だけ除外 | `cuda_cutlass_gemm_fused.cu:704, 1014` | **on** |

したがって p=511 / 767 / 1023 の `GEMM_FUSED` は §13 の 3 つの改修をすべて
載せた状態で測られてきた（§11・§12 と `p575_gap_study.md` §11 の全測定がこの上）。
**横展開すべき差分は無い。**

### 13.1 文字どおりの「z を 4 本の CUDA ストリームに」は天井がゼロ

念のため、z assembly を CUDA ストリームで分割する案の賞金を見積もると、
既存の 2 つの実測で天井が閉じている。

- **カーネル間の隙間が無い**。`p1023_gap_study.md` §13 の nsys GPU trace で、
  隙間は stage 周期の **0.02%（0.03 ms / 190 ms）**しかない。ストリームを増やして
  取れるのはこの隙間だけである。
- **z は 1 本で SM を埋めている**。§12.7 の ncu では z
  （`GemmBatchedDqdtAssembly`）が **SM スループット 91.52%**、残りの 3.5 ポイントは
  §11.4 で lift そのもの（138 µs = 1.11%）と同定済みで、数値契約上必須。
  p=511 は `Ne=1` なので z のバッチ数は 1 で、grid は
  `(nq2/64) x (Nq/32) = 4096 x 16 = 65536` CTA ある。SM は 148 枚なので、
  ストリームを 4 本に割っても各カーネルが 16384 CTA を持つだけで、
  **同時に走れる CTA は変わらない**。得られるのは並行性ではなく、
  ローンチ 3 回分と L2 再利用の劣化である。

**天井が測定誤差以下なので実装しない**（`AGENTS.md`「アブレーションで天井が
測定誤差以下」）。負けの方向に働く機構がはっきりしているぶん、
これは「効かない」ではなく「符号が負」である。

## 14. p≥511 に融合経路（`FUSED` / `FUSED_TC`）が無い理由（2026-09-03、資源の数え上げ）

`TODO.md` §5 の 1 件目。p=511 / 575 / 767 / 1023 の 4 本の gap study は
`CUDAFORTRAN_FUSED_TC` にも CUDA-core 融合にも一度も言及しておらず、
[`README.md`](README.md) の表注も「p≥511 は `GEMM` / `GEMM_FUSED` のみ」と
**事実**を書くだけで、**書かない判断の検討記録が無かった**。本節がその記録である。

測定は新規に取っていない。使ったのはすべて既存の実測値と、本節のために
login node で取り直した ptxas / `cudaGetDeviceProperties` の数え上げである
（commit は本節を追加したもの、親 `709334a`、GPU は RIKYU GB200、
`nvcc -arch=sm_100`、NVIDIA HPC SDK 26.3）。

### 14.1 ハードウェアの数え上げ（`cudaGetDeviceProperties`、GB200）

| 量 | 値 |
|---|---:|
| SM 数 | **152** |
| shared / SM | 233,472 B（228 KiB） |
| shared / block（opt-in 上限） | **232,448 B（227 KiB）** |
| レジスタ / SM | 65,536 |
| 最大スレッド / SM | 2,048 |
| `gridDim.x` 上限 | 2,147,483,647 |

（[`README.md`](README.md) の p=511 §13.1 は SM 数を 148 と書いているが、
この GPU が返すのは 152 である。本節の占有率はすべて 152 で計算した。
どちらでも下の結論は動かない。）

### 14.2 要素常駐型（p=7 … p=127 の `FUSED` / `FUSED_TC`）は不可能

低次数の融合カーネルは「1 ブロック = 1 要素」で、少なくとも 1 枚の
`Nq × Nq` 平面を shared に置く。

| Nq | 1 平面 = `Nq²·8` B | 227 KiB 上限に対して |
|---:|---:|---:|
| 128 | 131,072 | 0.56× |
| 256 | 524,288 | **2.26×** |
| 512 | 2,097,152 | **9.02×** |
| 1024 | 8,388,608 | **36.1×** |

**Nq=256 の時点で 2.26 倍あふれている。** これが p=255 の `FUSED_TC` が
要素常駐をやめてタイル型（§14.3）になっている理由であり、
p≥511 でも同じ理由で要素常駐は書けない。

さらに p≥511 は `Ne=1` なので、「1 ブロック = 1 要素」の grid は
**1 ブロック**である。152 SM のうち 1 枚しか使わない。仮に shared が
足りたとしても、この幾何では機械の 0.66% しか動かせない。

**したがって p≤127 の融合カーネルの実装様式は、p≥511 では
数え上げの段階で不可能である。**

### 14.3 タイル型（p=255 の `tendency_p255_kernel`）は資源としては**そのまま載る**

p=255 の `FUSED_TC` は 64×64 の出力タイル・`BK=16` の縮約チャンクを持つ
GEMM 形のカーネル 3 本（DIR=0/1/2）で、要素常駐ではない。ptxas
（`nvcc -O3 -std=c++17 -arch=sm_100 -Xptxas=-v`、本節で取り直した）:

```
tendency_p255_kernel<DIR, UseTc=true, ...>
  Used 168 registers, 0 bytes spill, 32768 bytes smem
```

- shared = `2·(BM+BN)·BK·8` = `2·128·16·8` = **32,768 B**
- スレッド = `TH255` = **128**、`__launch_bounds__(128, 3)`
- ブロック / SM = `65536 / (168·128)` = **3**（律速はレジスタ。shared なら 7 枚入る）
- 占有率 = `3·128 / 2048` = **18.75%**

**この 4 つはどれも `Nq` を含まない。** `Nq` が入るのはグリッドと
チャンク数だけである:

| p | Nq | ブロック数 `Nq³/(64·64)·Ne` | `gridDim.x` 上限比 | チャンク数 `Nq/BK` | wave 数 `blocks/(152·3)` |
|---:|---:|---:|---:|---:|---:|
| 255 | 256 | 4,096 | 0.0002% | 16 | 9.0 |
| 511 | 512 | 32,768 | 0.0015% | 32 | 71.9 |
| 575 | 576 | 46,656 | 0.0022% | 36 | 102.3 |
| 767 | 768 | 110,592 | 0.0051% | 48 | 242.5 |
| 1023 | 1024 | 262,144 | 0.012% | 64 | 574.9 |

メモリも足りる。融合経路の path 転送量は **64.5 B/node** で、
`GEMM_FUSED` の 145.3 B/node より少ない（[`README.md`](README.md)
「path 側の B/node」）。配列本数も融合の方が少ない（中間の
`flux_x/y/z` を持たない）ので、`GEMM_FUSED` が収まっている
p=1023（実測 peak 176.4 GiB）の内側に必ず収まる。

**結論: 数え上げは「不可能」とは言っていない。** タイル型は
1 ブロックあたりの資源を 1 バイトも増やさずに Nq=1024 まで伸びる。
唯一の実装上の注意は 32-bit 添字で、Nq=1024 では `3·Np = 3.22e9` が
`2^31−1` を越えるため、`Escale` の方向オフセットを 64-bit 化する必要がある
（`GEMM` 側で `p1023_gap_study.md` §1 が既にやったのと同じ修正）。
`tendency_p255_kernel` は `NQ255` を 6 か所でしか読んでおらず、
`MTILES` / `NTILES` / `blocks_per_elem` はそこから導いているので、
`NQ` のテンプレート化そのものは小さい。

### 14.4 したがって理由は資源ではなく**賞金**である。賞金は測ってある

p≥511 で融合が `GEMM_FUSED` から**削れる仕事は 1 つしかない**。

- **assembly / lift は既に融合済み**。`GEMM_FUSED` の z が最終の体積重み付けと
  6 面 lift をエピローグで持っている（`AGENTS.md` の経路定義）。
- **`elembnd_flux` は既に隠れている**。p=511 で 0.099 ms（§11.2）、
  p=1023 で 0.32 ms、どちらも side stream で完全に隠れている
  （[`p1023_gap_study.md`](p1023_gap_study.md) §13.2）。
- **カーネル間の隙間は 0.02%**（同 §13.2）。
- **`rk_update` は tendency の外**で、どの経路も同じだけ払う。

残るのは `volume_flux_kernel` だけである。融合カーネルは `q·u` を
mainloop のステージング中に作るので、この 1 本を丸ごと吸収できる。
その stage 比が**融合の賞金の天井**である:

| p | `volume_flux` [ms/stage] | stage [ms] | **天井** | 出所 |
|---:|---:|---:|---:|---|
| 511 | 1.036 | 12.483 | **8.30%** | §11.2（nsys job `71126`） |
| 575 | — | — | 7.4%（外挿） | 下のモデル |
| 767 | — | — | 5.5%（外挿） | 同 |
| 1023 | 8.311 | 186.06 | **4.47%** | [`p1023_gap_study.md`](p1023_gap_study.md) §13.2（nsys job `74738`） |

外挿のモデルは 1 行である。`volume_flux` は DRAM 律速で
時間 ∝ `Np`、3 本の GEMM は演算律速で時間 ∝ `Np·Nq` なので、
**天井 ∝ 1/Nq**。`8.30% × 512/Nq` は p=1023 で 4.15% を出し、
実測 4.47% と 1.08 倍で合う。

**天井は次数とともに 1/Nq で消える。** これが「p≥511 に融合経路が無い」
ことの本当の理由であって、shared でもレジスタでも占有率でもない。

### 14.5 損益分岐は「ピークの何 %」で書ける

融合経路の 3 本のカーネルが `GEMM_FUSED` の stage 時間を下回る条件は、
`FLOP / (stage · 40.1 TFLOP/s) ≥` 現行 `GEMM_FUSED` の対ピーク比、
すなわち **損益分岐効率 = [`README.md`](README.md) の `vs 40.1` 列そのもの**である。

| p | 損益分岐（= 現行 `GEMM_FUSED` の対ピーク比） | `FUSED_TC` の実績（現行値） |
|---:|---:|---:|
| 7 | 2.4% | **12.8%** |
| 15 | 8.7% | **19.0%** |
| 31 | 16.6% | **25.4%** |
| 63 | 30.1% | **41.6%** |
| 127 | 49.0% | **55.9%** |
| 255 | 67.6% | **73.7%** |
| 511 | **82.9–83.4%** | 未実装（最高実績は p=255 の 73.7%） |
| 575 | **85.6%** | 未実装 |
| 767 | **86.6%** | 未実装 |
| 1023 | **88.0%** | 未実装 |

`FUSED_TC` のリードは 10.4 → 10.3 → 8.8 → 11.5 → **6.9 → 6.1** ポイントと
高次数側で縮んでいる。p=511 にそのまま 6 ポイント足すと 88.9%、
時間で 11.64 ms/stage、`GEMM_FUSED` 比 **−6.8%** —— §14.4 の天井 8.30% の
内側で、2 つの独立な見積もりが p=511 で同じ桁に落ちる。

比較対象の強さも数えておく。p=511 の 3 本の GEMM は演算下限
（`2·Nq⁴/40.1e12` = 3.427 ms/本）の **88.7% / 93.2% / 94.6%**（§11.2）、
p=1023 の 3 本は合計で **ピークの 95.3%**
（[`p1023_gap_study.md`](p1023_gap_study.md) §13.2）。融合を除いた残りは
「手書き mainloop 対 ピークの 88.7–95.3% にいる CUTLASS」という賭けであって、
**融合の話ではない**。

### 14.6 判断

- **p=575 / 767 / 1023: 書かない、と決める。** 融合が削れる仕事の天井は
  5.5% / 4.5% で、しかも p=1023 では `volume_flux` を critical path から
  丸ごと外す不正アブレーションが **0.10% しか返さない**ことが実測されている
  （[`p1023_gap_study.md`](p1023_gap_study.md) §13.3。x/y GEMM が SM を
  96–97% 使っているので、DRAM 律速の flux は同時実行しても発行スロットを
  取れない）。吸収は重ねとは違うので天井は 4.47% の方だが、いずれにせよ
  損益分岐 86.6–88.0% は `FUSED_TC` の最高実績 73.7% より 13–14 ポイント上で、
  **カーネル 1 族を新設して取りにいく側の賞金ではない**。
- **p=511: 天井 8.30%、損益分岐 82.9–83.4%、外挿 −6.8%。決めずに開いたまま
  残す。** 数え上げでは実装可能（§14.3）で、賞金も測定誤差より 1 桁大きい。
  `AGENTS.md`「残る利得が小さいことは止める理由にならない」に照らすと、
  ここで「やらない」と書く根拠は無い。**未着手として `TODO.md` に立てる**のが
  正しい扱いで、本節はその見積もりである。手順は
  (1) `tendency_p255_kernel` を `NQ` でテンプレート化（`NQ255` は 6 か所）、
  (2) Fortran 側のゲート（`mod_advect3d_eq.f90` の次数チェックと
  `cuda_cal_dqdt_fused_tc`）、(3) `namelists/perf_p511_fused_tc.conf`、
  (4) 点変化係数で `CUDAFORTRAN_GEMM` 対照の全 134,217,728 点検証。
  Nq≥1024 に伸ばす場合だけ `Escale` 方向オフセットの 64-bit 化が要る。
- **CUDA-core 融合（`FUSED`）は 4 次数とも書かない。** p=255 CC は
  §26.6 の改良後でも **ピークの 42.7%**（1525.0 µs/stage）で、
  p=511 の損益分岐 82.9% の半分にも届かない。差は倍以上あり、
  外挿の不確かさで埋まる幅ではない。

**本節で採択したソース変更は無い。**

## 15. p=511 に融合経路を書いて測った（2026-09-03、`FUSED_TC` +5.46% で最速ではない）

§14.6 が「決めずに開いたまま残す」と書いた 1 件を、**実際に書いて測って**
閉じる。結論を先に書く。

- **§14.1–§14.4 の数え上げは現行ツリーでそのまま成り立った。** タイル型は
  資源を 1 バイトも増やさずに `Nq=512` に載り（168 レジスタ / spill 0 /
  32,768 B shared は `Nq=256` と**同一**）、賞金 `volume_flux` の stage 比も
  **8.36%**（§14.4 の 8.30%）で当たった。
- **§14.5 の外挿（リード 6 ポイント → −6.8%）は外れた。** 実測は
  `FUSED_TC` **13,084.0 µs/stage** 対 `GEMM_FUSED` **12,406.7 µs/stage** で
  **+5.46%（負け）**。対ピーク比は 79.1% で、損益分岐 83.4% を **4.3 ポイント
  下回る**。
- **§14.6 の判断は、p=575/767/1023 の「書かない」は維持、p=511 の
  「未着手として残す」は本節で閉じる。** 経路は書けたし数値も通ったが、
  **p=511 の最速は `GEMM_FUSED` のまま**である。

### 15.1 測定条件と物差し

- 親 commit **`e9d1037`**（`feature/cuda`）＋本節の作業ツリー。GB200 1 GPU、
  `make CUDA=1 GPUFLAGS=-gpu=cc100`（`cuda_dg_kernels_tc.cu` は
  `nvcc -O3 -std=c++17 -arch=sm_100`）、NVIDIA HPC SDK 26.3
- 入力は `namelists/perf_p511_{gemm,gemm_cute,gemm_fused,fused_tc}.conf`。
  **4 本は `DqdtKernel_Type` 以外まったく同一**（`Ne=1`、`PolyOrder=511`、
  `nstep=20`、`WarmupStep=2`、`dt=1.0D-8`、graph off、`8x8x4`）。
  `perf_p511_gemm_cute.conf` / `perf_p511_fused_tc.conf` /
  `perf_p511_fused_dfma.conf` は本節で追加した
- 検証は `namelists/val_p511_{gemm,fused_tc,fused_dfma}.conf`（`nstep=1`）
- **物差しは実行ファイルが出す `CUDA device *` ÷ 54**（= 18 measured step × 3
  stage）。経路ごとのラベルは `CUDA device GEMM tendency` / `... GEMM CUTE` /
  `... GEMM fused` / `... fused tendency` で、いずれも tendency を event で
  挟んだ device 時間である。`Cal_tend` や `Step loop per stage` とは別物
- 採否 A/B は **Slurm で占有した GPU 上の 12 ラウンド交互**、job **`78411`**
  （node **`c182`**）。カーネル内訳は同 job の nsys（`nstep=4`、
  `--resolve-symbols=false`、`DEBUGINFOD_URLS=` 空）
- 掃引・検証は login node

### 15.2 HEAD のベースライン（§9.6 の 3 値は今も再現する）

login 3-run 中央値（凍結 `scale-dg_extraction.p511_base`）:

| 経路 | device [s] | µs/stage | §9.6 の当時値 |
|---|---:|---:|---:|
| `CUDAFORTRAN_GEMM` | 0.718089 | **13,297.9** | 13,299.9 |
| `CUDAFORTRAN_GEMM_CUTE` | 0.724220 | **13,411.5** | 13,410.9 |
| `CUDAFORTRAN_GEMM_FUSED` | 0.671616 | **12,437.3** | 12,437.3 |

**3 経路とも §9.6 と 0.02% 以内**で、`3d4045c` の周辺カーネル融合と
volume GEMM の別ストリーム化は p=511 の**この物差しでは動いていない**
（`3d4045c` の p=511 `GEMM` −0.222% は別ジョブの占有 GPU A/B なので、
本表と直接は比べない）。比較対象は動いていないので、§14 の数え上げの
分母はそのまま使える。

### 15.3 実装したもの —— `tendency_p255_kernel` の `NQ` テンプレート化

§14.6 の手順見積もりのとおりで、見積もりより小さかった。

1. `cuda_dg_kernels_tc.cu`: `p255_epilogue` と `tendency_p255_kernel` と
   `p255_set_smem` に **`int NQ` テンプレート引数**を足し、`NQ255` 直参照
   （§14.3 が数えた 6 か所）を `NQ` にした。`MTILES` / `NTILES` /
   `blocks_per_elem` はそこから導かれるので自動で従う。
2. 同ファイル: `launch_tendency_dir_p255_impl` を
   `launch_tendency_dir_p255_nq<UseTc, NQ>` の薄い実行時ディスパッチにし、
   `extern "C"` の 2 本（`_tc` / `_dfma`）に **`int nq` 引数**を足した。
   未使用の `launch_tendency_xyz_p255_{tc,dfma}` は `NQ255` 固定のまま。
3. `mod_cuda_dg_kernels.cuf`: 上記 2 本の interface に `nq` を足し、
   `CUDA_P511_NQ = 512` を追加して
   `cuda_cal_dqdt_fused_p255_{tc,dfma}` のゲートを `Nq ∈ {256, 512}` にした。
4. `mod_advect3d_eq.f90`: `FUSED_TC` / `FUSED_DFMA` の次数ゲートに
   `Np == 512**3` を足し、p≥511 の dispatch guard で **p=511 に限って**
   この 2 経路を許可し、`fused_flux_bnd` の割り付け条件と `Nq == 512` の
   分岐を足した。`FUSED`（CC）は触っていない。
5. namelist 5 本（上記）。

**32-bit 添字は `Nq=512` では足りる。** 最大は `Escale` の方向オフセットで
`node + 2*npoint` = `3·Np − 1` = **4.03e8** < `2^31−1`。§14.3 が書いたとおり、
64-bit 化が要るのは `Nq=1024` からである。

**ptxas は §14.3 の数え上げを 1 桁も外していない**
（`nvcc -arch=sm_100 -Xptxas=-v`）:

| インスタンス | レジスタ | spill | shared |
|---|---:|---:|---:|
| `tendency_p255_kernel<DIR, 256, UseTc=true, *>` | 168 | 0 | 32,768 B |
| `tendency_p255_kernel<DIR, 512, UseTc=true, *>` | **168** | **0** | **32,768 B** |

`UseTc=false`（`FUSED_DFMA`）側は `NQ=256` / `512` の両方で 56–80 B の
cumulative stack を出す。これは `p255_gap_study.md` §26「副産物 2」が
`NQ=256` で記録したものと同じで、**次数を変えても変わらない**。

`Nq=512` で動くのは §14.3 が予告した 2 つだけである: grid が
4,096 → **32,768** ブロック（`gridDim.x` 上限の 0.0015%）、縮約チャンクが
16 → **32**。

### 15.4 数値検証（全 134,217,728 点、`SCALE_DG_VARYING_COEFF=1`）

owned `dqdt(:,1:Ne)` の全点を unformatted ではなく `ES24.16` の
テキストダンプで落として全点比較した（`SCALE_DG_DUMP_DQDT`）。
参照は `CUDAFORTRAN_GEMM`（p=511 に `SPLIT` は無い。§4.1 と同じ）。

| 比較 | 点数 | 最大絶対差 | 最大相対差 |
|---|---:|---:|---:|
| `FUSED_TC` 対 `GEMM` | 134,217,728 | **3.552714e-15** | 1.13e-09 |
| `FUSED_DFMA` 対 `GEMM` | 134,217,728 | **3.552714e-15** | 1.13e-09 |
| `FUSED_TC` 対 `FUSED_DFMA` | 134,217,728 | **全点ビット一致**（`cmp`） | — |

最大絶対差は §4.2 が `GEMM` 対 `GEMM_FUSED` で記録した 3.5527e-15 と
**同じ値**である。最大相対差が 1e-9 級なのは値が 0 に近い点で、これも §4.2 と
同型。`FUSED_TC` と `FUSED_DFMA` が全点ビット一致なのは、両者が同一ソースの
`UseTc` 違いで内積以外が完全に同じだからで、`AGENTS.md` の iso-schedule 規定を
`Nq=512` でも満たしていることの確認になっている。

**回帰**: `Nq=256` 側は本改修の前後で `perf_p255_fused_tc.conf` の owned
`dqdt` が**全点ビット一致**、時間も login 2-run で 881.5 → 882.0 µs/stage
（+0.05%、ばらつきの内側）。テンプレート化は `NQ=256` の生成コードを
変えていない。

### 15.5 占有 GPU A/B（job `78411`、node `c182`、12 ラウンド交互）

| 経路 | 中央値 µs/stage | レンジ | 対 `GEMM_FUSED` | 対ピーク（40.1 TFLOP/s） |
|---|---:|---:|---:|---:|
| **`CUDAFORTRAN_GEMM_FUSED`** | **12,406.7** | 12,404.2–12,408.7 | — | **83.4%** |
| `CUDAFORTRAN_FUSED_TC` | 13,084.0 | 13,082.0–13,089.8 | **+5.46%** | **79.1%** |
| `CUDAFORTRAN_GEMM` | 13,264.7 | 13,258.0–13,268.9 | +6.92% | 78.0% |
| `CUDAFORTRAN_GEMM_CUTE` | 13,382.1 | 13,376.6–13,384.4 | +7.86% | 77.3% |

**4 経路ともレンジ非重複。** アルゴリズム FLOP は
[`README.md`](README.md) の p=511 行の **4.150e11 FLOP/stage**。

読みどころは 2 つある。

- **`FUSED_TC` は `GEMM_FUSED` に 5.46% 負ける。** これが本節の答えである。
- **しかし `GEMM` に −1.36%、`GEMM_CUTE` に −2.23% 勝つ。** 融合が無い
  ライブラリ経路には勝てる、という位置にはいる。負けているのは
  **融合済みの CUTLASS 経路にだけ**である。

`FUSED_DFMA` は A/B に入れていない（login 3-run のみ、
0.708731 / 1.638270 s → **13,124.6 / 30,338.3 µs/stage**）。
**機構比 A（`FUSED_TC` / `FUSED_DFMA`）は 2.312×** で、
`p255_gap_study.md` §25 の p=255 2.278× の隣に置ける 7 次数目になる。
ただし**これは login 測定なので、§25 の 6 次数表（同一ジョブ交互）と
同じ表に入れてはならない**。

> **追記（2026-09-03、`reports/measurement_inventory.md` §7.3、job `78678`、
> node `c186`、tree `232dd27`）: この制限は解除された。** 5 経路を
> 1 ジョブ・12 ラウンド交互（全レンジ非重複）で取り直し、
> `FUSED_TC` **13 100.8** 対 `FUSED_DFMA` **30 330.7** µs/stage ＝
> **機構比 A = 2.315×**。上の login 値 2.312× と 0.13% 差で、
> **`p255_gap_study.md` §25 の 6 次数表の隣に置ける 7 次数目になった。**
> 同ジョブの他の 3 経路も §15.5 を別ノードで再現している:
> `GEMM_FUSED` 12 418.6（§15.5 は 12 406.7）、`GEMM` 13 284.2（13 264.7）、
> `GEMM_CUTE` 13 395.3（13 382.1）。`FUSED_TC` の相対位置は
> `GEMM_FUSED` に **+5.49%**（§15.5 +5.46%）、`GEMM` に **−1.38%**（−1.36%）、
> `GEMM_CUTE` に **−2.20%**（−2.23%）で、**3 つとも 0.03 ポイント以内で一致する。**
> 上の表と数字は当時の測定としてそのまま残す。

### 15.6 機構 —— 賞金は当たっていた。外れたのは mainloop の値段

nsys（同 job `78411`、`nstep=4`、`c182`、12 インスタンスの中央値）で
1 stage を全部割り付ける。

**`CUDAFORTRAN_FUSED_TC`:**

| カーネル | 時間 [ms] |
|---|---:|
| `tendency_p255_kernel<2(z), 512, TC>` | 4.2893 |
| `tendency_p255_kernel<0(x), 512, TC>` | 4.2755 |
| `tendency_p255_kernel<1(y), 512, TC>` | 4.2709 |
| **3 本合計** | **12.8357** |
| `elembnd_flux_group_kernel`（side stream） | 0.0392 |

**`CUDAFORTRAN_GEMM_FUSED`:**

| カーネル | 時間 [ms] |
|---|---:|
| z `GemmBatchedDqdtAssembly` 64×32（融合エピローグ） | 3.8658 |
| x CUTLASS `GemmBatched` 64×64 | 3.6711 |
| y `GemmBatchedScaleAdd` 64×64 | 3.5877 |
| **volume GEMM 3 本合計** | **11.1246** |
| `volume_flux_kernel` | **1.0376** |
| `elembnd_flux_kernel`（side stream） | 0.0843 |
| **3 GEMM + flux** | **12.1622** |

ここから帰属が閉じる。

| 項目 | ms/stage | `GEMM_FUSED` の 12.1622 に対して |
|---|---:|---:|
| 融合が**吸収した** `volume_flux` | **−1.0376** | **−8.53%** |
| 手書き mainloop が**払った超過**（12.8357 − 11.1246） | **+1.7111** | **+14.07%** |
| 差引 | **+0.6735** | **+5.54%** |

実測の wall 差は **+5.46%**（§15.5）で、**0.1 ポイント以内で一致する**。
1 stage の残りは 2 経路で同じである（`rk_update` 0.586+0.615 ms は tendency の
外、`elembnd_flux` はどちらも side stream、halo 0.020 ms）。

**したがって §14.4 の賞金の数え上げは正しかった。** `volume_flux` は
1.0376 ms、stage 12.407 に対し **8.36%**（§14.4 の 8.30%、nsys job `71126`）。
`1/Nq` モデルも生きている。

**外れたのは §14.5 の「リードを 6 ポイント足す」という外挿の側である。**
理由は 1 行で書ける ——
**手書き mainloop は mma 屋根の 80% で頭打ちだが、CUTLASS は 89–95% にいる。**

| | 演算下限 `2·Nq⁴/40.1e12` | 実測 | 屋根に対して |
|---|---:|---:|---:|
| `FUSED_TC` z / x / y | 3.427 ms/本 | 4.2893 / 4.2755 / 4.2709 | **79.9 / 80.2 / 80.2%** |
| `GEMM_FUSED` z / x / y | 3.427 ms/本 | 3.8658 / 3.6711 / 3.5877 | **88.6 / 93.3 / 95.5%** |

（`FUSED_TC` の 3 本は flux とエピローグも中でやっているので、80% は
**GEMM 分の FLOP だけを分子にした下限側の値**である。それでも
CUTLASS との差は 9–15 ポイントあり、賞金 8.4% では埋まらない。
`GEMM_FUSED` の 88.6 / 93.3 / 95.5% は §11.2 の 88.7 / 93.2 / 94.3% を
別ジョブで再現している。）

**符号が変わった機構**を、6 次数の実績と並べて書く。

| p | 損益分岐（`GEMM_FUSED` の対ピーク比） | `FUSED_TC` の対ピーク比 | リード [pt] |
|---:|---:|---:|---:|
| 7 | 2.4% | 12.8% | +10.4 |
| 15 | 8.7% | 19.0% | +10.3 |
| 31 | 16.6% | 25.4% | +8.8 |
| 63 | 30.1% | 41.6% | +11.5 |
| 127 | 49.0% | 55.9% | +6.9 |
| 255 | 67.6% | 73.7% | +6.1 |
| **511** | **83.4%（実測）** | **79.1%（実測）** | **−4.3** |

**リードはゼロを Nq=256 と Nq=512 の間で横切る。** §14.5 は最後の 2 点
（+6.9 → +6.1）を「縮んでいる」と正しく読みながら、外挿では**縮まないもの**
として扱ってしまった。両側の増分を並べると読み違えの中身が見える:
`FUSED_TC` は 55.9 → 73.7 → **79.1**（+17.8 → **+5.2**）と**減速**し、
損益分岐は 49.0 → 67.6 → **83.4**（+18.6 → **+15.8**）と**減速していない**。
手書き mainloop は自分の屋根（80%）に着いてしまい、CUTLASS はまだ伸びる。
この 2 本の曲線の交差が、賞金 8.4% の内側か外側かを決めていた。

**（2026-09-03 追記: この未説明は閉じた。）** 本節が
「`Nq=256` で同じ 3 本 + `volume_flux` の nsys 分解を取っていないので、
屋根が次数によらず一定なのかは測っていない」と書いた 1 件を、
`p255_gap_study.md` §29 が **`Nq=256` の nsys 分解**（tree `44b02a6`、
Slurm job `78530`、node `c187`、`namelists/perf_p255_fused_tc.conf` と
`DqdtKernel_Type` だけ違うその複製）で閉じた。**答えは「一定ではない」**:

| | 3 本平均 Nq=256（§29.4） | 3 本平均 Nq=512（上表） | 増分 |
|---|---:|---:|---:|
| 手書き mainloop（`FUSED_TC`） | **75.2%** | 80.1% | +4.9 pt |
| CUTLASS（`GEMM_FUSED`） | **83.1%** | 92.5% | +9.4 pt |
| CUTLASS − 手書き | **+7.9 pt** | **+12.4 pt** | +4.5 pt |

したがって上の「**手書き mainloop は mma 屋根の 80% で頭打ち**」という
1 行は**訂正する**。80% は Nq=512 で観測した 1 点であって屋根ではなく、
両方の曲線が Nq とともに上がり、**手書きの方が上がり方が遅い**。
本節の結論（リードが Nq=256 と 512 の間でゼロを横切る）は変わらないが、
その機構は「片方が屋根に着いた」ではなく **「賞金が `1/Nq` で縮み
（14.16% → 8.36%）、効率差が広がる（7.9 → 12.4 pt）2 本の曲線の交差」**
である。帰属は両次数とも実測 wall 差と 1 ポイント以内で閉じている
（Nq=256 は −5.46% 対 実測 −6.36%、Nq=512 は +5.54% 対 実測 +5.46%）。
上の測定表は書き換えていない。なお §29.6 のとおり、6 次数リード表の
p=255 行は現行ツリーでは **69.3 / 73.9 ＝ +4.7 pt**（本節当時は +6.1 pt）で、
交差点は本節を書いた時点より Nq=256 側に寄っている。

### 15.7 判定 —— §14 の帰趨

**§14.1 / §14.2 / §14.3（資源の数え上げ）: 維持。** ptxas が 168 レジスタ /
spill 0 / 32,768 B を `NQ=512` でも返し、実際に動いて全 134,217,728 点が
通った。「タイル型はそのまま載る」は**実装で確認された**。

**§14.4（賞金 = `volume_flux` 1 本、stage 比 8.30%）: 維持。** 現行ツリーの
nsys で **8.36%**。`1/Nq` モデルも動いていない。

**§14.5（損益分岐 82.9–83.4%）: 維持。** 実測 **83.4%**。
**ただし同節の「リードを 6 ポイント足すと −6.8%」という外挿は訂正する。**
正しい読みは §15.6 の表で、リードは Nq=512 で **−4.3 ポイント**である。
訂正の理由は「リードを次数に依らない定数として外挿した」ことであり、
**外挿すべきだったのは 2 本の効率曲線それぞれ**だった。

**§14.6（判断）:**

- **「p=575 / 767 / 1023 は書かない」は維持し、根拠が外挿から実測に変わった。**
  §14.6 はこれを「損益分岐 86.6–88.0% は実績 73.7% の 13–14 ポイント上」で
  決めていた。本節はその実績を **Nq=512 で 79.1%** と実測し、**同じ Nq で
  損益分岐が 83.4%** であることも実測した。融合側が既に 1 段手前で負けて
  いる以上、損益分岐がさらに 3–5 ポイント上がる 575 / 767 / 1023 で
  勝つ道は無い。**この 3 次数は引き続き書かない。**
- **「p=511 は決めずに残す」は本節で閉じる。書いて、測って、負けた。**
  `AGENTS.md`「残る利得が小さいことは止める理由にならない」に従って
  実装したのは正しかった —— 止める根拠が無かったのは事実で、
  **その根拠を作ったのが本節の測定である**。`p511` の最速経路は
  **`CUDAFORTRAN_GEMM_FUSED` のまま**で、README の最速経路表は動かない。
- **CUDA-core 融合（`FUSED`）は p=511 では書かない。** §14.6 の判断を維持する。
  **本節は CC を p=511 で測っていない**（`cuda_dg_kernels_fused_highp.cu` は
  触っていない）。判断は数え上げで、しかも今回強くなった: p=255 CC は
  `p255_gap_study.md` §28.10 の改良後で **ピークの 45.2%**（1440.0 µs/stage、
  §14.6 が引いた 42.7% から上がっている）だが、p=511 の損益分岐 **83.4%** は
  その **1.85 倍**である。しかも CC は**どの次数でも TC に勝ったことがない**
  （主比 B は p=255 で 1.630×）ので、TC が 79.1% で 4.3 ポイント負けている
  次数で CC が勝つ経路は無い。**これは測定ではなく数え上げであることを
  明記しておく。**

### 15.8 `NQ` テンプレート化の扱い —— production dispatch に残す

**残す。** `PolyOrder=511` で `DqdtKernel_Type = "CUDAFORTRAN_FUSED_TC"` /
`"CUDAFORTRAN_FUSED_DFMA"` を指定すると、この 2 本が起動する。既定オフの
ノブにはしていない。理由:

- **役割の定義を満たしている。** `AGENTS.md` の `CUDAFORTRAN_FUSED_TC` は
  「`cuda_dg_kernels_tc.cu` の Tensor Core 融合カーネル、`UseTc=true`、
  `mma.sync.m8n8k4`」であり、p=511 の実体は**同一ソースの同一
  インスタンス**（テンプレート引数 `NQ` だけが違う）である。
  `FUSED_DFMA` も `UseTc=false` の同一ソースで、内積以外で乖離していない
  （§15.4 で全点ビット一致）。**新しい次数に既存経路を開くことは経路定義の
  変更ではない**ので、`AGENTS.md` の編集は要らない（本節は編集していない）。
- **数値コントラクトを満たしている。** §15.4。
- **モデルにとって必要な行である。** p=511 は本節まで `GEMM` 系だけの行で、
  「融合経路が無い」ことの根拠が外挿だった。実測の融合経路がある方が、
  次数横断の効率曲線（§15.6 の表）が 7 点で引ける。
- **他次数への副作用が無い。** `NQ=256` は生成コードもビット一致も時間も
  不変（§15.4 の回帰）。
- **最速経路表は動かさない。** p=511 の最速は `GEMM_FUSED`。
  `p255_gap_study.md` §28.10 が p=255 CC について書いたのと同じ扱いで、
  **経路が存在することと最速であることは別**である。

これは役割外アブレーションではない（役割内の、負けた本番経路である）ので、
`AGENTS.md` が要求する「ラベルして production dispatch から外す」扱いには
当たらない。

### 15.9 到達点と、閉じた項目

- **p=511 に `CUDAFORTRAN_FUSED_TC` / `CUDAFORTRAN_FUSED_DFMA` が開通した。**
  13,084.0 / 13,124.6（login）/ 30,338.3 µs/stage。
- **p=511 の最速は `CUDAFORTRAN_GEMM_FUSED` 12,406.7 µs/stage のまま。**
  融合経路は +5.46%（レンジ非重複）。
- **`TODO.md` §5 の「p≥511 に融合経路が存在しない理由」は本節で完全に閉じた。**
  §14 が 4 次数のうち 3 つを数え上げで閉じ、残した 1 つ（p=511）を
  本節が測定で閉じた。
- **本節で採択した性能上の変更は無い。** 追加したのは経路の開通と
  namelist だけである。
