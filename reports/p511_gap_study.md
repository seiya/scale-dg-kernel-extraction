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
