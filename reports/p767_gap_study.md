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

### 1.1 なぜ p=767 か（2026-09-03 追記）

**この次数は GPU デバイスメモリ容量から選ばれている。** 対象としたのは
Wisteria/BDEC-01 の **40 GiB** と Tsubame の **94 GiB** の 2 つで、
**p=767 は 94 GiB に収まり、GEMM に都合がよく、かつ p=575 と p=1023 の
中間的な値**として選ばれた。（出所は本リポジトリの作業者。本節を書くまで
レポートにもコミットメッセージにも記録が無かった。）

数え合わせ: 現行 tree の主要配列は `144*Np` = **60.75 GiB**（下表と §3 の訂正）で、
`p1023_gap_study.md` が全次数で確認した事前見積もり `payload*1.25 + 2 GiB` を当てると
**77.9 GiB** となり 94 GiB に収まる（40 GiB には収まらない）。
`Nq = 768 = 64*12` は 64 の倍数なので cuBLAS / CUTLASS の 64 幅タイルが割り切れる
（「GEMM に都合がよい」の内容はここだと思われるが、作業者の言明に含まれていないので
推測である）。中間性については `Nq` が 576 < **768** < 1024 の位置にある。

他の次数が `Nq` を 2 冪（8, 16, 32, 64, 128, 256, 512, 1024）で揃えているのに対し
p=575 / p=767 だけが 2 冪でないのは、この容量基準で選ばれたためである。
p=575 の選定理由は [`p575_gap_study.md`](p575_gap_study.md) §1.1 を見よ。

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

## 8. Ozaki Scheme I / II（2026-08-28 追記）

p=511 と同様に Ozaki 経路を開放し、同一 DOF（`Ne=1`）で計測。
commit `38952e4`、`nstep=20`、`WarmupStep=2`、
`OzakiSliceCount=8` / `OzakiModuliCount=14`、GB200 login ノード。

| 経路 | µs/stage | native GEMM 比 |
|---|---:|---:|
| `CUDAFORTRAN_GEMM` | **188.4 ms** | 1.00 |
| `CUDAFORTRAN_GEMM_OZAKI1` | **160.6 ms** | **0.85×** |
| `CUDAFORTRAN_GEMM_OZAKI2` | **434.1 ms** | **2.3×** |

**Scheme I が native volume GEMM を上回る最初の次数**（device 160.6 対 188.4 ms/stage）。
ただし最速は `GEMM_FUSED`（60.4 ms/stage）のままで、Ozaki は volume GEMM 置換の
参考比較にとどまる。

> **（訂正 2026-09-03）** 上表は commit `38952e4` の測定で、そのツリーの
> `EmulationMantissaControl` の既定は **DYNAMIC** だった（`fd091fc`、2026-08-29 で
> FIXED に変わる）。`gemm_assignment_and_carrier.md` §9.8.5 が確定させたとおり
> **DYNAMIC の `OZAKI1` は DG の要求精度を満たさない**（p=7 の owned `dqdt` で
> 3.04e-01）ので、**上表の `OZAKI1` の比は無効な精度設定での速度である。**
> 精度を満たす FIXED で測り直すと **`OZAKI1` = 618.8 ms/stage ＝ native の 9.78×**、`OZAKI2` = 82.1 ms/stage ＝ 1.30×
> （job `78846`、node `c186`、既定 FIXED 55 bit / 7 slices、
> [`measurement_inventory.md`](measurement_inventory.md) §10.5）。
> `OZAKI2` は FIXED / ADP・7 / 14 moduli のどれでも数値が合わない（同 §9.8.5）ので、
> **`OZAKI2` の行はどの設定でも「その精度での速度」である。**
> **本表は当時の DYNAMIC 既定での測定として書き換えずに残す。**

> **（訂正 2026-08-29）** 上表の native **188.4 ms** は 1 step の device 合計で、
> RK 3 stage で割っていない。1 stage は約 62.8 ms で、§5 の GEMM 62.817 ms/stage
> と整合する。Ozaki 比はそのまま読める。Scheme I が volume GEMM を上回る、という
> 結論は比が同じなので変わらない。

## 9. cuBLAS FP64 emulation（2026-08-28 追記）

`CublasEmulation = .true.`、`CUDAFORTRAN_GEMM` のみ。commit `a1cdb57`、
§5 と同条件（`nstep=30`、`WarmupStep=5`）、GB200 login ノード、3 run 中央値。

| 経路 | device [ms/stage] | native GEMM 比 |
|---|---:|---:|
| `CUDAFORTRAN_GEMM`（native） | 62.82 | 1.00 |
| `CUDAFORTRAN_GEMM` + emulation | **122.03** | **1.94×** |

入力は `input_p767_val_gemm_emu.conf`。
[`cublas_emulation_survey.md`](cublas_emulation_survey.md) §4 参照。

## 10. 2026-08-29 の経路横断について

p=7…255 は同一実行ファイルで再測定したが、**p=767 は再実行していない**。
[`README.md`](README.md) まとめ表の p=767 行は本レポートの
`GEMM` 62.817 / `GEMM_FUSED` 60.362 ms/stage のまま。

## 11. `GEMM_FUSED` の残り天井を測って探索終了（2026-09-01）

コミット: `f7788bb`（コード変更なし。候補は測って全部戻した）。
GPU: RIKYU GB200 1 枚。入力は `namelists/perf_p767_gemm_fused.conf`
（`Ne=1`、`PolyOrder=767`、`nstep=30`、`WarmupStep=5`、graph off）。
nsys job `71131`、ncu job `71132`（c178、`--set full`、nstep=4）。
採否の A/B は job `73625`（flux_yz 重ね、c179）と job `74704`（flux CTA 128、c399）。

p=767 の最速は `CUDAFORTRAN_GEMM_FUSED` のまま。採用ゼロ。

### 11.1 ベースライン

login GPU 0、同一 conf を 4 回。device fused は 4.46030 / 4.46076 / 4.60724 /
4.45768 s（75 stage）。外れ値 4.607 を除く中央値 **4.46030 s = 59.471 ms/stage**、
Main **182.31 ms/step**。§5 の 60.362 ms/stage より 1.5% 短いが、当時の表は
書き換えない。理論 volume GEMM は `6*Nq^4 / 40.1e12 = 52.05 ms`。device 全体の
アルゴリズム FLOP 2.097e12 を 59.471 ms で割ると **35.26 TFLOP/s（ピークの 87.9%）**。

### 11.2 律速

nsys（12 launch 中央値、µs）:

| カーネル | µs | 1 方向 mma 下限 17350 µs に対する比 |
|---|---:|---:|
| z assembly | 18927 | 91.7%（epilogue 込み） |
| x scale | 18227 | 95.0% |
| y scaleadd | 18027 | 96.2% |
| `volume_flux` | 3504 | — |
| `elembnd`（side） | 183 | x/y の裏に隠れている |

ncu（クロック固定。時間は採否に使わない）:

- **y**: SM 96.77%、DRAM 5.39%、占有率 18.4%（レジスタ 168、3 CTA）。stall の 47% は math pipe。**演算パイプが律速。**
- **z**: SM 93.90%、DRAM 5.33%、L2 hit 84%、占有率 12.5%（レジスタ 254、4 CTA）。y より 0.90 ms 長い差の半分は lift（§11.4）。**これも演算パイプ。flux_z の 24 回読みは L2 が捕まえている。**
- **`volume_flux`**: DRAM 91.64%（7.27 TB/s）、long scoreboard が stall の 81%。**帯域律速。** 100% までの天井は 0.29 ms（stage の 0.49%）。
- x は ncu の `-c 6` から外れた。nsys では y と 1% 以内。

1 文: **3 本の GEMM は FP64/Tensor パイプ、flux は DRAM、elembnd は隠れている。**

### 11.3 候補と結果

| 候補 | 天井 | 結果 | 機構 |
|---|---|---|---|
| C++ `double4` の volume_flux | DRAM 屋根まで 0.29 ms | login 3/3 で device **+0.08%**（4.464 vs 4.460）。戻した | バイトは減らない。p=7 TC のベクトル化が DRAM 飽和で負けたのと同じ |
| flux CTA 256→128 | 同上 0.29 ms | job `74704` の A 4.44897–4.44938、B 4.44943–4.44980。**レンジ重複、差が無い**。戻した | 占有率理論値はどちらも 100%。DRAM 91% では CTA 幅が動かない |
| `flux_y/z` を x GEMM の side2 に重ねる | yz を消すと最大 ~2 ms | job `73625` で A 4.456、B 4.540（**+1.89%**、3 対ともレンジ非重複）。戻した | x は SM 97%。フルグリッドの DRAM カーネルが GEMM を直列化する。`p63_gap_study.md` §26 の +5.9% と同じ向きで、ここでは x が 18 ms あっても負けが消えない |
| z タイル 64×32→64×64（warp 32×64、`Nq>=512`） | z を y の 96.8% SM に寄せる 0.6 ms | login 3/3 で device **+20%**（5.36 vs 4.46）。戻した | ncu の L2 84% が正しく、N を広げても DRAM は減らない。shared 増とワープ形状で mainloop が壊れる。p=255 §10.4 の 64×64 負けが、A が L2 に載らない次数でも成り立つ |
| lift を消すアブレーション（不正） | — | login 中央値 4.424 vs 4.460、**−0.80%（0.48 ms/stage）** | z−y の 0.90 ms のうち半分。契約内で lift を消す手は残っていない（p=255 §10.8） |

範囲外（実装せず）: RK 更新の z epilogue 融合（p=255 で +5.5%）、速度の代表スカラー化、`volume_flux` を書かないこと。

### 11.4 終了条件

残る契約内の天井は (1) GEMM をピークの 100% まで寄せる約 1.7 ms（y は既に 96.8% で、CUTLASS 2.x のタイルは p=255 と本節で掃引済み）、(2) flux の DRAM 屋根 0.29 ms で、測った 2 形はゼロか負け、(3) lift 0.48 ms で取り方が無い。

**p=767 `CUDAFORTRAN_GEMM_FUSED` の探索を終了する。** 最速経路は変わらない。

## 12. 融合エピローグの accumulator repad と `GEMM_CUTE` の開通（2026-09-01）

コミット: 本節を追加したコミット（親は `acdbd8a`）。GPU は RIKYU GB200 1 枚、
`make CUDA=1 GPUFLAGS=-gpu=cc100`。入力は `namelists/perf_p767_gemm_fused.conf`
（`Ne=1`、`PolyOrder=767`、`nstep=30`、`WarmupStep=5`、graph off）。
採否の A/B は占有 GPU job `74732` / `74820`（ともに c182、12 回交互）、
横展開は job `74821`（c185、10 回交互）。ncu も同一ジョブ内で採った。

§11 で探索を終了したが、`feature/cuda` の `09cb3b3` が**バッチ launcher にだけ**
`RepadEpilogue` を入れていたため、融合経路の x と y が素の epilogue のまま
取り残されていた。本節はそれを塞ぎ、ついでに p≥511 で塞がっていた未融合対照
`GEMM_CUTE` を開いた。**§11.4 の「探索終了」は本節で訂正する。**

### 12.1 rebase の影響はゼロ

`f7788bb` 凍結バイナリと `acdbd8a` ビルドを login で 3 対交互: 4.48551–4.49332 対
4.48849–4.49206 s。**レンジ重複＝差なし**。§11.1 の 4.46030 s との差は login GPU の
日差であり、`acdbd8a` までの共有コード変更は p=767 を動かしていない。本節の分母は
job `74820` の base 中央値 **4.48084 s = 59.744 ms/stage** である。

### 12.2 採用: x と y の epilogue を 8 パディングする（−0.127%）

z の assembly epilogue は `RepadEpilogue<...,8>` を最初から使い、`09cb3b3` で
バッチ launcher にも入った。だが融合経路の x は `run_gemm_nn_scaled`（device 版
`Gemm`）、y は `run_volume_gemm_y_scaleadd` を通るのでどちらも素のままだった。
y は `Kernel` の epilogue 型を差し替えるだけ、x は device 版 operator が
epilogue を差し替えられないので `run_gemm_nn_scaled_repad` で Params を手組みした
（列優先 C の device `Gemm` は転置問題を回すので、`to_underlying_arguments` から
組む）。

job `74820`（c182、12 回交互、`Cal_tend` [s/75 stage]）:

| variant | 中央値 | range | base 比 |
|---|---:|---:|---:|
| base | 4.48084 | 4.48058–4.48109 | — |
| y のみ repad | 4.47949 | 4.47921–4.47969 | **−0.030%** |
| x + y repad | **4.47515** | 4.47490–4.47539 | **−0.127%** |

3 群は互いにレンジ非重複。`Main` と `CUDA device GEMM fused` も同符号・同幅。
**59.744 → 59.669 ms/stage。**

機構は同一ジョブの ncu（job `74732` が base と y のみ、`74820` が x+y。
shared store bank conflict の総数）:

| kernel | base | x+y repad | 命令数 |
|---|---:|---:|---|
| x (`kernel::Gemm`, `GemmXScale`) | 32.57 M | **4.26 M（−86.9%）** | 4,695,515,136（不変） |
| y (`GemmBatchedScaleAdd`) | 33.59 M | **4.15 M（−87.6%）** | 5,659,213,824（不変） |
| z (`GemmBatchedDqdtAssembly`、元から repad 済み) | 5.03 M | 5.00 M | 7,041,171,456（不変） |

**命令数が 1 命令も動かず conflict だけが消える。** ncu duration も x 33.923 →
33.782 ms、y 33.921 → 33.833 ms と同符号である。`p127_gap_study.md` §20.2 は
shared-store conflict を −72.6% にしても壁時間が動かなかった例だが、あれは
同時に命令が **+11.2%** 増えていた。ここは命令が完全に不変なので、ncu の既知
バイアス（`SKILL.md` 手順 3）に乗らずに占有 GPU の A/B が同符号で出る。

効果が p=7 の −4.11%（`09cb3b3`）に対して p=767 で −0.127% なのは、epilogue が
mainloop の **1/K** で希釈されるからである。K=768 は p=7 の 96 倍あり、比は桁で合う。

### 12.3 横展開（job `74821`、c185、10 回交互、`Cal_tend` 中央値）

| 次数 | base | x+y repad | 判定 |
|---|---:|---:|---|
| p=7 (`Ne=32³`) | 0.10424 | 0.10421 | レンジ重複 = **差なし** |
| p=15 (`Ne=16³`) | 0.04061 | 0.04060 | レンジ重複 = **差なし** |
| p=31 (`Ne=8³`) | 0.46974 | 0.46989 | レンジ重複 = **差なし** |
| p=127 (`Ne=2³`) | 0.21631 | **0.21525** | レンジ非重複、**−0.487%**（728.3 → 724.7 µs/stage、`Cal_tend`/297 stage） |

x の repad は `Nq>64` の枝にしか無く（`Nq<=64` は x が cuBLAS）、y の repad は
全次数に入る。`p255_gap_study.md` §10.4 の「(2)(3) を `Nq<=64` の枝に持ち込むと
ptxas のコード生成が悪化して p=63 が +2.7%」という前例があるので Nq ゲートを
警戒したが、**Nq≤64 の 3 次数とも回帰しない**ので次数で分けない。p=127 の取り分が
p=767 の 3.8 倍なのは K=128 で希釈が 6 分の 1 だからで、これも 1/K と整合する。

### 12.4 タイル掃引はすべて負け（login 3 回、中央値）

> **（訂正 2026-09-01）** 本節が掃引したのは当時の x（単発 `64x128`）と y
> （`64x64` 4 段）である。`p575_gap_study.md` §11 / `p511_gap_study.md` §12 で
> `Nq>=512` の x は `64x64` batched に、x/y とも 3 段になった。§14.2 のとおり
> p=767 では両形に差が無いので、**タイルでは取れない**という結論は変わらない。

`p255_gap_study.md` §10.4 のタイル掃引は Nq=256 のものなので、K が 3 倍の
p=767 で測り直した。天井は x が mma 屋根に対して 0.88 ms（1.5%）、y が 0.68 ms
（1.1%）。融合経路が使う `GemmXScale` / `GemmYScale` だけを差し替えている
（`GEMM_CUTE` の `GemmX` / `GemmY` は触っていない）。

| variant | `Cal_tend` [s] | base 比 |
|---|---:|---:|
| base（x 64×128 s3 / y 64×64 s4） | 4.48481 | — |
| y 128×64 s3（warp 64×32） | 4.50288 | +0.40% |
| y 64×128 s3（warp 32×64） | 4.50948 | +0.55% |
| y 64×64 s5 | 4.52302 | +0.85% |
| x 128×128 s3 | 4.53190 | +1.05% |
| x 64×256 s3 | 4.53882 | +1.20% |

5 形とも base のレンジ 4.48379–4.48554 と非重複の負け。**p=255 の掃引結果は
K が 3 倍でもそのまま成立し、x と y の残り天井はタイルでは取れない。**
5 形とも p=127 の全点比較は max abs 1.77636e-15 で通っている。

### 12.5 `GEMM_CUTE` を p≥511 で開いた —— 融合の値段が初めて測れる

`mod_advect3d_eq.f90` の次数ゲートが p=511/575/767/1023 で `GEMM_CUTE` を
弾いていた。`GEMM_CUTE` は `GEMM_FUSED` の**未融合対照**（`AGENTS.md`）で
volume GEMM タイルを共有するので、ここが塞がっていると融合エピローグの値段が
測れない。ゲートに 1 行足すだけで p=767 は初回から完走した。

> **（訂正 2026-09-01）** 本節の数値は login ノードで採ったもので、同じノードで
> 別セッションが GPU を使っていた。占有 GPU で採り直した §13.1 の
> `GEMM_CUTE` 62.970 / `GEMM_FUSED` 59.652 ms/stage（融合 **−5.27%**、
> volume GEMM 55.141 対 55.658）を正とする。融合の値段という結論は変わらない。

login 3 回の中央値（`Cal_tend` と、両経路が出す volume GEMM 区間）:

| 経路 | volume GEMM [ms/stage] | 残り [ms/stage] | `Cal_tend` [ms/stage] |
|---|---:|---:|---:|
| `CUDAFORTRAN_GEMM_CUTE`（未融合、**初測定**） | 55.323 | 7.826 | 63.149 |
| `CUDAFORTRAN_GEMM_FUSED`（本節） | 55.749 | 3.994 | **59.743** |

**融合は GEMM 側に +0.426 ms/stage（+0.77%）払って、GEMM 外を
7.826 → 3.994 ms/stage（−3.83 ms）に減らしている。** 差引 −3.41 ms/stage
（**−5.4%**）。§5 の cuBLAS `GEMM` 62.82 との比較では「CUTLASS 化の分」と
「融合の分」が混ざっていたが、これで分離できた。

同時に **p=767 の純 GEMM の床**も出た: 55.323 ms/stage は 1 方向 mma 下限
17.35 ms の 3 倍 = 52.05 ms に対して **94.1%** で、融合版の 55.749 は 93.4%。
`GEMM_FUSED` の GEMM 外に残る 3.994 ms は `volume_flux` 3.504 ms（§11.2、
DRAM 91.6%）とその他 0.49 ms であり、収支が閉じる。

### 12.6 数値検証

- p=127（`Ne=2³`、`SCALE_DG_VARYING_COEFF=1`）で `CUDAFORTRAN_SPLIT` と全
  16,777,216 点比較: base / y repad / x+y repad / タイル 5 形すべて
  **max abs 1.77636e-15、max rel 2.22029e-16**（base と同値）。
- p=767 実寸（`Ne=1`、`SCALE_DG_VARYING_COEFF=1`）で、独立実装の
  `CUDAFORTRAN_GEMM`（cuBLAS）を参照に全 **452,984,832 点**を比較した
  （`namelists/val_p767_gemm.conf` / `val_p767_gemm_fused.conf`、ダンプは 1 本
  11.3 GB）:

  | 経路 | max abs | max rel |
  |---|---:|---:|
  | `CUDAFORTRAN_GEMM_FUSED`（x+y repad） | **1.77636e-15** | 2.22017e-16 |
  | `CUDAFORTRAN_GEMM_CUTE`（本節で開通） | **0（ビット一致）** | 0 |

  §4 の 3.55e-15 と同オーダーで、GEMM 縮約順の丸め差と整合する。3 経路とも
  最終 min/max は `-9.999676920720005E-01` / `9.999676926585226E-01` で一致。
  通常の benchmark 係数でも `val_p767_gemm_fused.conf` の min/max は §4 の
  公表値 `-9.999676850160175E-01` / `9.999676855992745E-01` のまま。

### 12.7 残っているもの

§11.4 の 3 項のうち (1) は本節で 0.075 ms 取り、タイルでは取れないことを
測って確定した。残りは §11 と同じで、(2) flux の DRAM 屋根 0.29 ms、
(3) lift 0.48 ms、(1) の残り約 3.1 ms（GEMM を屋根まで寄せる分。§11.2 の nsys で
x 95.0% / y 96.2% / z 91.7%、§12.5 の純 GEMM 合計で 94.1%）。範囲外は
§11.3 のとおり。**最速は
`CUDAFORTRAN_GEMM_FUSED` のまま。**

## 13. 残り 3 天井を占有 GPU で測って探索終了（2026-09-01）

> **【物差しの注記（2026-09-03 追加）】本節の `ms/stage` はすべて `Cal_tend`
> 系（`Cal_tend` ÷ 75 stage）であり、[§15](#15-測定衛生-cal_tend-の-per-stage-は実行長に依存する2026-09-01)
> がこれを訂正している。** `Cal_tend` はホスト時計で GPU 同期を持たないため
> **ちょうど 1 stage 分を取りこぼす**（§15.3 / §15.5）。75 stage では偏り
> −1.3%、33 stage では −3.0%。したがって
> **§13.1 の 59.652 / 62.592 / 62.970 を、他次数や他レポートの
> `CUDA device *` 系の値と同じ表に並べてはならない。**
> p=767 `GEMM_FUSED` の正しい絶対値は **テンダンシー 58.592 ms/stage**（§15.6）。
> **本節の A/B 判定・順位・パーセントはいずれも同一 conf・同一 stage 数の
> 比較なので、偏りは両群に共通で差し引かれ、結論は 1 つも動かない**（§15.4）。
> 現行ツリーでの絶対値は `measurement_inventory.md` §7.4（job `78678`、
> `CUDA device *` ÷ 33 = 60 294.4 µs/stage）を参照。

コミット: 本節を追加したコミット（親は `2bf8c49`。コード変更なし。候補は測って
全部戻した）。GPU は RIKYU GB200 1 枚、占有 GPU job `74865`（c179、8 構成を
10 回交互）。入力は `namelists/perf_p767_gemm_{fused,cute}.conf` と
`DqdtKernel_Type` / `CutlassMmaShape` だけを変えた一時 conf。

§12 を書いた時点で、**測っていない天井が 3 つ**残っていた。本節でそれを潰す。
なお §12.4 / §12.5 は login ノードで測っており、同じノードで別セッションが GPU を
使っていたことが後から判明した。差が 5% 級の §12.5 の結論は動かないが、
本節の数値の方が信頼できる（§12.5 に訂正注記を置いた）。

### 13.1 結果

`Cal_tend` [s/75 stage]、中央値、レンジ:

| 構成 | 中央値 | range | ms/stage | `GEMM_FUSED` 比 |
|---|---:|---:|---:|---:|
| `CUDAFORTRAN_GEMM`（cuBLAS、未融合） | 4.69438 | 4.69389–4.69485 | 62.592 | +4.93% |
| `CUDAFORTRAN_GEMM_CUTE`（CUTLASS、未融合） | 4.72272 | 4.72242–4.75678 | 62.970 | +5.56% |
| **`CUDAFORTRAN_GEMM_FUSED`（現行）** | **4.47391** | 4.47368–4.76976 | **59.652** | — |
| mma `16x8x4` | 4.48069 | 4.48041–4.48086 | 59.743 | +0.15% |
| mma `16x8x8` | 4.57029 | 4.56969–4.66012 | 60.937 | +2.15% |
| mma `16x8x16`（`TileK=32`） | 7.86467 | 7.86141–7.87031 | 104.862 | **+75.8%** |
| cuBLAS x + weighted y（`Nq=64` 枝を p=767 へ） | 4.49732 | 4.49689–4.99049 | 59.964 | +0.52% |
| z assembly epilogue 全消去（**不正**） | 4.42973 | 4.42933–4.43010 | 59.063 | **−0.99%** |

volume GEMM 区間だけ: `GEMM_CUTE` 4.13558（**55.141 ms/stage**）、
`GEMM_FUSED` 4.17431（55.658）。融合の代金は GEMM 側 **+0.517 ms**、
`Cal_tend` 全体では 62.970 → 59.652 で **−5.27%**（§12.5 の login 値 −5.4% を置換）。

### 13.2 天井 1: CUTLASS mainloop は cuBLAS に 0.60% 負けている

> **（訂正 2026-09-01）** `p575_gap_study.md` §11.2 の batched x が入った後は
> **0.45%**（§14.3）。縮んだが残る。結論は変わらない。

同じ未融合ドライバ・同じ epilogue・同じ `separable_lift_assembly` で違うのは
volume GEMM のライブラリだけなので、**62.592 対 62.970 の 0.378 ms/stage
（0.60%）が mainloop 差そのもの**である。§12.4 でタイル 5 形が全部負けたことを
「屋根に当たっている」と読むのは誤りで、**屋根まで 0.6% ある**。

契約内で取れるかを測った。`Nq<=64` の枝が持つ cuBLAS-x 分岐（cuBLAS x ＋
`Ey*acc + Ex*Dx` の y epilogue）を p=767 に伸ばすと **+0.523%**（レンジ非重複、
p=127 全点比較 3.55271e-15）。`p255_gap_study.md` §10.4 が Nq=256 で測った +0.7% と
同符号で、K が 3 倍でも成立する。機構は §10.3(1) の裏返しで、x を cuBLAS にすると
`Escale_x` の前送りが消えて z が 5 テンソルに戻り、y は `escale_x` を余分に読む。
y と z は契約上 CUTLASS 固定（`AGENTS.md`: 融合 z を cuBLAS/CUTLASS z + 別 lift に
分解してはならない）なので、**この 0.60% は融合契約の代金であって、契約内では
取れない**。これは負の結果として公開する価値がある。

### 13.3 天井 2: `CutlassMmaShape` は 8x8x4 が最良（変更ではなく確認）

「sm_100 では 4 形とも `DMMA.8x8x4` に落ちるので速くなりようがない」という
`cuda_cutlass_gemm_fused.cu` の記述は命令については正しいが、**`16x8x16` だけは
`TileK=32` を要求してタイル形状そのものが変わる**（`WarpShape::kK /
InstShape::kK >= 2`、`mma_base.h:128,132`）ので、p=767 で測る価値があった。
結果は **+75.8%**。shared が CTA あたり倍になり占有率が落ちる代金が、K 段数が
半分になる利得を大きく上回る。`16x8x4` は +0.15%（レンジ非重複の小さな負け）、
`16x8x8` は +2.15%。**4 形の順位が p=767 でも `sm90_mma_shape_survey.md` と同じ**
であることを確認した。

### 13.4 天井 3: z epilogue は lift 以外に 0.19% しか無い

z の assembly epilogue をまるごと消し、素の z GEMM を `dqdt` に書くだけにした
不正アブレーション（lift も weighting も `Dx`/`Dy`/`Ez` の読みも消える）は
**−0.987%（0.589 ms/stage）**。§11.3 の lift だけ消したアブレーションが
**−0.80%（0.48 ms）**だったので、**lift 以外の epilogue 仕事はすべて合わせて
0.11 ms = 0.19%** しかない。`p255_gap_study.md` §10.2 は同じアブレーションが
Nq=256 で −8.0% だったが、K が 3 倍になると epilogue は 1/K で希釈されるので
桁が合う。lift は契約内で消せない（§11.3、`p255_gap_study.md` §10.8）。

### 13.5 終了条件

p=767 `GEMM_FUSED` の 59.652 ms/stage に対して、残る天井は全部測った:

| 残り | 天井 | 状態 |
|---|---:|---|
| CUTLASS mainloop 対 cuBLAS | 0.60% | 契約内の唯一の手（cuBLAS x）が **+0.52%**。取れない |
| lift | 0.80% | 契約内で消す手が無い（§11.3） |
| `volume_flux` の DRAM 屋根 | 0.49% | `double4` +0.08%、CTA 128 差なし（§11.3） |
| z epilogue の lift 以外 | **0.19%** | §13.4 で新たに測った下限。手が無い |
| タイル / stage / mma shape | — | 5 形 +0.40〜+1.20%（§12.4）、mma 3 形 +0.15〜+75.8%（§13.3） |

**p=767 `CUDAFORTRAN_GEMM_FUSED` の探索を 59.652 ms/stage で終了する。**
最速経路は変わらない。§12 で取った −0.127% が、このラウンドの唯一の採用である。

## 14. `Nq>=512` 分岐の後始末と再測定（2026-09-01）

> **【物差しの注記（2026-09-03 追加）】本節の `ms/stage` はすべて `Cal_tend`
> ÷ 33 stage であり、[§15](#15-測定衛生-cal_tend-の-per-stage-は実行長に依存する2026-09-01)
> がこれを訂正している。** 33 stage での `Cal_tend` の偏りは **−3.0%** なので、
> **§14.1 / §14.3 の 58.642 / 58.619 / 61.584 / 61.859 を横断比較の絶対値に
> 使ってはならない。** §14.5 の「p=767 `GEMM_FUSED` は 58.642 ms/stage」も
> 同じ物差しの値で、**正しい絶対値はテンダンシー 58.592 ms/stage**（§15.6）。
> **§14 の A/B（repad −0.075%、batched x のレンジ重複、融合 −5.20%、
> cuBLAS 差 0.45%）はいずれも同一 conf・同一 stage 数の比較なので有効**（§15.4）。
> なお §13（75 stage）と §14（33 stage）の絶対値が 1.77% 違うのは性能差ではなく
> **この偏りの差**である（§15.1）。現行ツリーの値は
> `measurement_inventory.md` §7.4。

コミット: 本節を追加したコミット（親は `b344c6d`）。GPU は RIKYU GB200 1 枚、
占有 GPU job `75114` / `75124`（ともに c179、12 回交互）。入力は
`namelists/perf_p767_gemm_fused.conf`（`Ne=1`、`nstep=15`、`WarmupStep=4`、
graph off）＝ **11 measured step × 3 = 33 stage**。`p575_gap_study.md` §11 が
この conf を導入したので本節以降はこれを使う。**§11–13 は `nstep=30` /
`WarmupStep=5`（75 stage）で測っており、絶対値は本節と直接比較できない。**

下の「追記」が書くとおり、`p511_gap_study.md` §12（融合 y の 3 段化）と
`p575_gap_study.md` §11.2 / §11.13（`Nq>=512` の x を y と同じ `64x64` batched に、
かつ 3 段共有）が p=767 にもそのまま載った。本節はその上で (a) 外れていた x の
repad を戻し、(b) §12.4 / §13.2 が古くなっていないかを測り直す。

### 14.1 採用: batched x にも accumulator repad を戻す（−0.075%）

`Nq>=512` の x は `run_gemm_batched_nn_scaled`（chunk ループ、CUTLASS 標準
epilogue）を通るようになったので、§12.2 で x に入れた repad が**この次数では
外れていた**（`run_gemm_nn_scaled_repad` は `64<Nq<512` 専用になった）。
`run_gemm_batched_nn_scaled_capped` を書いて、`run_gemm_batched_nn_capped` と
同じ形 —— full `batch_count` ＋ capped `grid.z` の 1 ローンチ ＋ pad 8 —— に
した（chunk ループは呼ばれなくなったので削除）。

job `75114`（`Cal_tend` [s/33 stage]）:

| variant | 中央値 | range | ms/stage | 比 |
|---|---:|---:|---:|---:|
| rebase 直後 | 1.93664 | 1.93623–1.93672 | 58.686 | — |
| **+ batched x の repad** | **1.93518** | 1.93484–1.93532 | **58.642** | **−0.075%** |

レンジ非重複。同一 job の ncu が機構を出す（x カーネル）:

| | rebase 直後 | repad |
|---|---:|---:|
| shared store conflict | 30.70 M | **2.54 M（−91.7%）** |
| `smsp__inst_executed.sum` | 5,440,684,032 | 5,440,684,032（**不変**） |
| ncu duration | 34.189 ms | 34.088 ms |

§12.2 と同じで、**命令が 1 つも動かず conflict だけが消える**。

### 14.2 `Nq>=512` の batched x は p=767 では差が無い

`p575_gap_study.md` §11.2 は「単発 `64x128` の x は FP64 ピークの 84%、同じ FLOP の
y（`64x64` batched）は 95%」を根拠に x を batched にし、p=575 で **−2.94%** を得た。
p=767 でその取り分を切り分けるため、分岐を無効化して単発 `64x128`（repad 込み）に
戻した版と A/B した（job `75124`）:

| variant | 中央値 [s/33 stage] | range | ms/stage |
|---|---:|---:|---:|
| 単発 `64x128` x | 1.93443 | 1.93417–1.93550 | 58.619 |
| batched `64x64` x（現行） | 1.93514 | 1.93485–1.93528 | 58.641 |

**レンジ重複＝差が無い。** p=575 の −2.94% は p=767 には出ない。§11.2 の ncu が
p=767 の単発 x を **mma 下限の 95.0%** と測っていたことと整合する（p=575 の 84% は
その次数固有）。**現行の分岐はそのまま残す**（p=575 の勝ちを壊さず、p=767 では
中立）。次数をまたぐ knob の効きが `Nq` で変わる例として記録する。

### 14.3 §12 / §13 の再測定

同 job `75114`、同 conf:

| 経路 | volume GEMM [ms/stage] | `Cal_tend` [ms/stage] |
|---|---:|---:|
| `CUDAFORTRAN_GEMM`（cuBLAS、未融合） | — | 61.584 |
| `CUDAFORTRAN_GEMM_CUTE`（CUTLASS、未融合） | 56.069 | 61.859 |
| `CUDAFORTRAN_GEMM_FUSED`（本節） | 56.640 | **58.642** |

- **融合の値段（§12.5 / §13.1 の再測）**: GEMM 側に **+0.571 ms/stage**、
  `Cal_tend` は 61.859 → 58.642 で **−5.20%**。§13.1 の −5.27% と同じ。
- **cuBLAS 差（§13.2 の再測）**: 61.859 − 61.584 = **0.275 ms/stage = 0.45%**。
  batched x が入って §13.2 の 0.60% から縮んだが**消えてはいない**。残るのは
  y と z の mainloop 差で、どちらも契約上 CUTLASS 固定なので取れない、という
  §13.2 の結論は向きも理由も変わらない。

### 14.4 数値検証

- p=511（`Ne=1`、`SCALE_DG_VARYING_COEFF=1`、全 134,217,728 点）: batched x の
  repad 版は **repad 無しとビット一致**、両者とも `CUDAFORTRAN_GEMM`（cuBLAS）に
  対し **1.77636e-15**。`Nq>=512` の分岐を通る次数での検証はこれ。
- p=127（`Ne=2³`、全 16,777,216 点、`CUDAFORTRAN_SPLIT` 基準）: **1.77636e-15**。

### 14.5 現在地

p=767 `GEMM_FUSED` は **58.642 ms/stage**（33 stage 基準）。§13.5 の天井一覧は
cuBLAS 差が 0.60% → 0.45% になった以外はそのままで、採用できる候補は残っていない。
最速は `CUDAFORTRAN_GEMM_FUSED` のまま。

## 15. 測定衛生: `Cal_tend` の per-stage は実行長に依存する（2026-09-01）

コミット: 本節を追加したコミット（コード変更なし）。GPU は RIKYU GB200 1 枚、
占有 GPU job `75159`（c386）/ `75616`（c163）/ `75625`（c179）/ `75626`（c358）。

§14 で conf が `nstep=30`/`WarmupStep=5` から `nstep=15`/`WarmupStep=4` に
変わった結果、同じバイナリの per-stage が 59.69 から 58.64 ms へ **1.77% 動いた**。
これはコードでもノード差でもなく、**`Cal_tend` の測り方の性質**だった。

### 15.1 コードか conf かの切り分け（job `75159`）

rebase 前 `89cd52c` と現行を、旧 conf・新 conf の両方で 10 回交互:

| | 旧 conf（75 stage） | 新 conf（33 stage） |
|---|---:|---:|
| rebase 前 | 59.694 | 58.640 |
| 現行 | 59.657 | 58.600 |

**conf を変えた効果 −1.766% / −1.772%**（2 バイナリで一致）、
**コードを変えた効果 −0.062% / −0.068%**（2 conf で一致）。

### 15.2 原因はクロック垂れではない（job `75616` / `75625`）

`WarmupStep` を 4 に固定して `nstep` だけ動かすと per-stage は動くが
（15 → 30 で **+1.86%**）、`nstep=30` のまま `WarmupStep` を 4 → 5 にしても
**+0.05%** しか動かない。warmup の切り方ではなく**実行長**が効く。

長い方の conf を走らせながら `nvidia-smi` を 200 ms 間隔で採ると（job `75625`）、
計算区間 5.4 秒を通して **SM クロックは 2062 MHz（= 最大）に張り付き**、電力は
900 W（上限 1200 W）で頭打ち、温度 47 → 56 °C、throttle reason は **0x0**。
**クロックも電力も温度も垂れていない。** 熱による説明は測定で否定された。

### 15.3 正体: `Cal_tend` に一定のオフセットがある（job `75626`）

`WarmupStep=4` を固定して `nstep` を 8 / 15 / 22 / 30 / 45 と変え、6 回ずつ:

| `nstep` | measured steps | `Cal_tend` [s] | `Cal_tend`/(3·steps) [ms/stage] | `Main`/step [s] | `Main`/(3) [ms/stage] |
|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 0.66515 | **55.430** | 0.18246 | 60.819 |
| 15 | 11 | 1.93865 | **58.747** | 0.18215 | 60.717 |
| 22 | 18 | 3.21202 | **59.482** | 0.18207 | 60.690 |
| 30 | 26 | 4.66741 | **59.839** | 0.18204 | 60.681 |
| 45 | 41 | 7.39626 | **60.132** | 0.18200 | 60.667 |

`Cal_tend`/(3·steps) は 55.4 から 60.1 まで **8.5% も動く**のに、
**区間ごとの増分は完全に一定**である:

| 区間 | `Cal_tend` 増分 | `Main` 増分 |
|---|---:|---:|
| step 5–11 | 60.643 ms/stage | 60.659 |
| step 12–18 | 60.637 | 60.646 |
| step 19–26 | 60.641 | 60.661 |
| step 27–41 | 60.641 | 60.644 |

5 点の線形回帰:

| | 傾き | 切片 |
|---|---:|---:|
| `Main` | 60.651 ms/stage | **+0.0021 s（ほぼ 0）** |
| `Cal_tend` | 60.640 ms/stage | **−0.0625 s** |

**傾きは両者一致し、`Cal_tend` だけが一定の −62.5 ms のオフセットを持つ。**
これは 1 stage 分（60.6 ms）にあたる。`Timer` は `system_clock` の純ホスト時計
（`mod_common.f90`）で GPU 同期を持たないので、最後の `Timer_stop` の後に
GPU 上で流れ切る 1 stage 分が積算から漏れる、という形と量が一致する。

**したがって `Cal_tend/(3·steps)` は実行長に依存する量であり、
絶対値の指標にしてはならない。** 偏りは `1/(3·steps)` で、
`nstep=15`/`WarmupStep=4`（33 stage）なら **−3.0%**、
`nstep=30`/`WarmupStep=5`（75 stage）なら **−1.3%**、
p=127 の `nstep=100`（297 stage）なら −0.34% である。
§14 の 58.642 と §13 の 59.652 は、**同じ真値 ≈ 60.5 ms/stage の
偏りの違う 2 つの見え方**であって、性能差ではない。

### 15.4 何が変わり、何が変わらないか

- **変わらない**: 本レポートの A/B 判定はすべて**同一 conf・同一 stage 数**で
  取っているので、オフセットは両群に共通で差し引かれる。パーセントは
  オフセットの分だけ相対的に 1〜3% 過大に出るだけで（−0.075% の真値は
  −0.073%）、採否も符号も動かない。レンジ非重複の判定も変わらない。
- **変わる**: 次数や conf をまたいだ**絶対値の比較**。`README.md` のまとめ表と
  各 gap study の per-stage 値は、いずれも `Cal_tend` 基準なら
  `1/(3·steps)` だけ低めに出ている。
- **推奨**: 絶対値は `Main per step`（切片ほぼ 0）か、`nstep` を 2 点以上振った
  **増分の傾き**で出す。`Cal_tend` の per-stage を載せるときは stage 数を併記する。
- なお `Cal_tend` の傾き 60.64 ms/stage は `Main`/3 と一致しており、
  **テンダンシー以外（RK 更新など、タイマ区間の外で enqueue された仕事）も
  含んでいる**。§11.2 の nsys のカーネル和 58.7 ms/stage ＋ RK 更新 2.3 ms と
  整合する。`Cal_tend` を「テンダンシーだけの device 時間」と読むのは正しくない。

### 15.5 5 つのタイマの偏りはすべて「ちょうど整数 stage」である

job `75626` の 5 点（12 / 33 / 54 / 78 / 123 stage）で、プログラムが出す
5 つの時間すべてを回帰した:

| 出力 | 傾き（真値） | 切片 | stage 換算 |
|---|---:|---:|---:|
| `Main` | 60.651 ms/stage | +2.1 ms | **±0** |
| `Cal_tend` | 60.640 | −62.5 ms | **−1.00 stage** |
| `Volume derivate + surface lift` | 60.640 | −62.5 ms | **−1.00 stage** |
| `CUDA device GEMM fused` | **58.592** | +58.7 ms | **+1.00 stage** |
| `FUSED volume GEMM only` | **55.080** | +55.1 ms | **+1.00 stage** |

偶然ではない。ホスト時計系（`Timer`）は最後の 1 stage を**取りこぼし**、
CUDA イベント系（`dg_ev_*`、二重バッファで 1 stage 遅れて harvest する）は
1 stage **余分に数える**。5 点とも切片がぴったり ±1 stage に乗る。

**したがって割る数を直すだけで、コードを触らずに正しい値が出る**:

| 出力 | 正しい割り方 |
|---|---|
| `Main` | ÷ (3·steps)（そのまま） |
| `Cal_tend` / `Volume derivate + surface lift` | ÷ **(3·steps − 1)** |
| `CUDA device GEMM fused` / `FUSED volume GEMM only` | ÷ **(3·steps + 1)** |

`nstep` を 2 点振って増分の傾きを採れば、オフセットを知らなくても消える。
**タイマ本体は直さない**。過去の全次数・全レポートの per-stage と基準が
1 stage 分ずれ、並行して走っている他次数の測定ともぶつかるためである。

### 15.6 p=767 の現在地（偏りを除いた値）

| | ms/stage |
|---|---:|
| 1 step 全体 ÷ 3（`Main`、偏り無し） | 60.651 |
| **テンダンシー（融合区間、CUDA イベント）** | **58.592** |
| volume GEMM のみ | **55.080** |

差の 60.651 − 58.592 = **2.06 ms/stage** は RK 更新と halo で、ncu の
`main_rk_update` 2.33 ms と整合する。§11.2 の nsys のカーネル和
58.7 ms/stage とも合う。

§13 の 59.652（75 stage）と §14 の 58.642（33 stage）は、どちらも
`Cal_tend` を `3·steps` で割った値で、`(3·steps − 1)` で割り直すと
**60.46 / 60.47** と一致する。**同じ実体の、偏りの違う 2 つの見え方**であって
性能差ではない。以降、本レポートで絶対値を書くときは
**テンダンシー 58.592 ms/stage** を使う。

## 追記（2026-09-01）: p=575 側から入った `Nq >= 512` の分岐

[`p575_gap_study.md`](p575_gap_study.md) §11 が volume GEMM に `Nq >= 512` の
分岐を 2 つ足したので、本次数にもそのまま載る。
本レポート本文の 60.362 ms/stage は 2026-08-28 の値で、`p511_gap_study.md` §12
の融合 y の 3 段化（本次数で −0.10%）より前である。

- **§11.2**: 融合 x を `64x128` の 1 本から、y と同じ `64x64` batched へ。
- **§11.13**: その x も `GemmYScaleShallow` を共有して 3 段パイプラインにする
  （`GEMM_CUTE` の x / y も同じ mainloop）。

両方入った実行ファイルと、batched x だけ入れて x/y は 4 段のままの実行ファイルを
占有 GPU 上で 8 回交互に測った（`p575_gap_study.md` §11.16、job `74975`、
`namelists/perf_p767_gemm_fused.conf`）:
**60383.3 → 60231.2 µs/stage（−0.252%）**。この A/B が測っているのは
**batched x の上で x/y を 4 段から 3 段にする効果**であって、
`p511_gap_study.md` §12 が融合 y 単独で測った値とは分母も対象も違う。
数値は p=575 で 191,102,976 点ビット一致（§11.13）。
