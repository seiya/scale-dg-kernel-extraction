# p=1023 GEMM / GEMM_FUSED 対応

## 1. 結論

`PolyOrder=1023`（`Nq=1024`, `Np=2^30`）を
`CUDAFORTRAN_GEMM` と `CUDAFORTRAN_GEMM_FUSED` で実行可能にした。
既存のruntime Nq対応GEMMに加え、32-bit境界を越える `Escale` 方向offsetと、
GB200 189471 MiBに収めるためのvolume-size一時配列を修正した。

点ごとに変化する `u`, `v`, `w`, `Escale`, `normal_fn`, `Fscale` を使い、
両経路の全 owned `dqdt` 1,073,741,824 点を比較した。最大絶対差は
`3.5527136788005009e-15`、非有限値は0で、差は浮動小数点丸めの範囲だった。

GB200 の同一入力を3回ずつ実行した中央値では、warm-up後のdevice tendencyが
GEMMの194.058 ms/stageに対してGEMM_FUSEDは187.617 ms/stageで、3.32%短い。

## 2. 測定条件

- 日付: 2026-08-28（Asia/Tokyo）
- ブランチ: `feature/p511`
- base commit: `47f41981b5c3e4ba412686101acabb7fb9ac3094`
- 測定状態: 本レポートと同時にコミットする working tree
- GPU: NVIDIA GB200、189471 MiB
- 性能測定GPU: index 1
- GPU UUID: `GPU-d5214545-6d82-2be9-a314-442682ff446b`
- driver: 580.173.02
- compiler: NVIDIA HPC SDK 26.3
- target: CUDA Fortran `cc100`、C++ CUDA `sm_100`
- Slurm job: なし（通常のlogin-node GPU実行。`nsys` / `ncu`は未使用）
- CUDA graph: off
- profiler: なし

数値検証はGPU index 0、性能測定は測定開始時に空きが最も多かったindex 1を使った。
いずれも同じGB200、driver、buildである。

ビルドコマンド:

```bash
module load nvhpc
make clean
make -j4 CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100 \
  CUTLASS_HOME=/data1/rkp00015/rku00044/scale-dg-kernel-extraction/third_party/cutlass
```

interface変更後、非CUDA buildも次でclean buildした。

```bash
make clean
make -j4
```

最終的なworking executableは上記CUDA buildである。

## 3. 境界修正

### 3.1 成立する32-bit index

`Ne=1`, `Nq=1024`では次となる。

| 項目 | 値 | signed 32-bit内 |
|---|---:|:---:|
| `Nq` | 1,024 | yes |
| `Nfp=Nq^2` | 1,048,576 | yes |
| `NfpTot=6*Nfp` | 6,291,456 | yes |
| `Np=Nq^3` | 1,073,741,824 | yes |
| 最大halo field index `Np+6*Nfp` | 1,080,033,280 | yes |
| y GEMM batch `Nq*Ne` | 1,024 | yes |

y GEMMのbatchはCUDA `grid.z <= 65535`制約を十分下回る。point index、face
index、VMap値、block数は32-bitのままでよく、全device kernelを64-bit化する
必要はない。

### 3.2 `Escale` の方向offset

`Escale(:,:,3)` の先頭offsetは `2*Np=2^31` elementsで、signed 32-bit境界を
越える。GEMM_FUSEDのC++ custom epilogueは既に `std::int64_t npoint` から
3方向のbase pointerを作っていた。

通常GEMMのassembly kernelは従来 `Escale(idx+2*npoint)` をdevice内で評価して
いたため、p=1023ではoverflowする。変更後はFortran array descriptorがnative
address kindで `Escale(1,1,1:3)` の3 base pointerを形成し、kernelへ別々に渡す。
kernel内は各pointerを32-bit `idx`だけで参照するので、既存次数へ64-bit device
index演算を追加しない。

## 4. メモリ削減

### 4.1 最初のOOM

従来の通常GEMMは主要配列が `160*Np = 160 GiB`だった。GB200の公称容量内に
見えるが、OpenACC/CUDA runtime、allocation管理、geometryとmappingも必要であり、
最後の8 GiB allocation時に空き約6 GiBでOOMになった。

調査すると、通常GEMMには実行時に読まれない `surface_lift` が残り、さらに
z-GEMMの `deriv_z` は直後のassemblyが各点を一度読んで `dqdt`へ上書きするだけ
だった。

### 4.2 採用した配置

- 通常GEMMから未使用の `surface_lift` allocationとinterfaceを除いた。
- z-GEMMを `beta=0` で `dqdt`へ直接出力した。
- assemblyは同じ点のz微分を `dqdt`からregisterへ読んだ後、最終値を同じ
  `dqdt`へ書く。

GEMMの浮動小数点演算、縮約順、global memoryのwrite/read回数は変わらない。
変わるのは同じz中間値を置くallocationの名前だけである。

p=1023、`Ne=1`の主要device allocationは両経路とも次となる。この表の144 GiBは
概算に便利な主要配列のpayloadであり、deviceの必要容量ではない。小さい演算子、
geometry、mapping、OpenACC allocatorの予約領域は含めない。

| 配列群 | bytes / `Np` | GiB |
|---|---:|---:|
| packed halo付き `q/u/v/w` | 64 | 64 |
| owned `q0/dqdt` | 16 | 16 |
| `Escale` | 24 | 24 |
| `volume_flux_x/y/z` | 24 | 24 |
| `volume_deriv_x/y` | 16 | 16 |
| 合計 | `144*Np` | **144** |

この配置で通常GEMMとGEMM_FUSEDがともに実機で完走した。

### 4.3 配列payloadと実使用量の校正

前節のような配列和だけでは必要容量を過小評価する。全device配列をbyte単位で
数えると、両経路のapplication-requested payloadは
`154,962,804,744 byte = 144.320 GiB`である。一方、CUDA `cudaMemGetInfo`を
初期化前、equation setup後、mainのOpenACC data配置後、最初のstep後、解放後に
記録すると次になった。GPU index 1、上記のvalidation input、`nstep=1`を使用した。

| 経路 | startup used | setup後 used | data配置後 peak used | startupからのpeak増分 | payloadとの差 | peak時free |
|---|---:|---:|---:|---:|---:|---:|
| `GEMM` | 0.842 GiB | 40.991 GiB | 177.259 GiB | **176.416 GiB** | **32.096 GiB** | 6.741 GiB |
| `GEMM_FUSED` | 0.878 GiB | 41.001 GiB | 177.236 GiB | **176.358 GiB** | **32.038 GiB** | 6.764 GiB |

初回step後の追加確保はなく、両経路の差は58 MiBで測定時の外部使用量の揺らぎと
同程度である。したがって、現在のGEMMとGEMM_FUSEDはpayloadだけでなく実際の
必要量も同じとみなせる。`acc end data`後もusedが約177 GiBのままだったが、process
終了後の`nvidia-smi`ではGPU 1は158 MiBへ戻った。これはリークではなく、NVHPCの
OpenACC allocatorがprocess内で解放済みblockをpoolに保持しているためである。

次数依存性を確認するため、同じGEMM allocation sequenceを追加で測定した。

| p | requested payload | startupからのpeak増分 | allocator/runtime差 | payload比 |
|---:|---:|---:|---:|---:|
| 511 | 18.080 GiB | 22.167 GiB | 4.087 GiB | 22.60% |
| 575 | 25.730 GiB | 30.105 GiB | 4.375 GiB | 17.00% |
| 767 | 60.930 GiB | 71.173 GiB | 10.243 GiB | 16.81% |
| 1023 | 144.320 GiB | 176.416 GiB | 32.096 GiB | 22.24% |

差は固定費ではなくallocation sizeとpool bucketに依存する。この4点では
`payload * 1.25`が実測peak増分をすべて上から覆う。そのため、事前判定は次を使う。

```text
required = exact_array_payload * 1.25 + 2 GiB safety
feasible = required <= cudaMemGetInfo startup free
```

p=1023では保守見積もりが182.400 GiB、実測開始時freeが183.158 GiBなのでfit、
実測peak増分は176.416 GiBだった。以前の主要配列だけの160 GiB、未使用surfaceを
消した後の152 GiBという値に同じallowanceを適用すると、それぞれ約200.4 GiB、
190.4 GiBとなり、実測startup freeを越える。従来配置がOOMし、現配置だけが通る
結果とも整合する。

再現用に`SCALE_DG_REPORT_DEVICE_MEMORY=1`を追加した。通常実行では呼ばれないため
性能への影響はない。出力を保存し、次でpayload内訳、実測差、余裕を同時に確認できる。

```bash
SCALE_DG_REPORT_DEVICE_MEMORY=1 ./scale-dg_extraction input_p1023_val_gemm.conf \
  | tee /tmp/p1023_memory.log
python3 estimate_device_memory.py --poly-order 1023 --path GEMM \
  --memory-log /tmp/p1023_memory.log
```

25%は今回のNVHPC 26.3、driver 580.173.02、同じallocation sequenceに対する経験的な
上限であり、compiler、driver、allocator mode、配列の確保順を変えた場合は再校正する。

## 5. 数値検証

### 5.1 方法

各経路の最初のRK stageが計算したowned `dqdt(:,1:Ne)`を一時ファイルへ出力し、
全1,073,741,824個のFP64値を比較した。両実行に
`SCALE_DG_VARYING_COEFF=1`を指定し、点変化する速度・幾何係数・面係数を使用した。

各dumpは1,073,741,824行、26,843,545,600 byteであることを確認した。固定幅
`ES24.16`を読む並列比較器で全点を集計し、約50 GiBのdumpと比較器は検証後に
削除した。

### 5.2 結果

| 係数 | 比較点数 | exact不一致 | 非有限値 | 最大絶対差 | 最大相対差 |
|---|---:|---:|---:|---:|---:|
| 点変化 | 1,073,741,824 | 468,585,417 | 0 | 3.5527136788005009e-15 | 4.6778976414349106e-9 |

最大相対差は値が0に近い点で生じる。最大絶対差は数ulpの範囲であり、cuBLASと
CUTLASSの異なる縮約順による丸め差と整合する。全owned fieldと全6面を含む
比較なので、定数速度や代表スカラーへの特殊化はない。

通常benchmark係数の1-step最終min/maxは両経路で
`-9.999992746461386E-01` / `9.999992747335267E-01`、点変化ケースは
`-9.999992815793513E-01` / `9.999992816672274E-01`で一致した。

回帰としてp=511でも点変化係数の両経路を1-step実行し、最終min/maxが
`-9.999738234621957E-01` / `9.999738239901469E-01`で一致した。

## 6. 性能

### 6.1 入力と方法

測定用一時入力は `/tmp/conf_perf_p1023_gemm.conf` と
`/tmp/conf_perf_p1023_gemm_fused.conf` とし、違いは `DqdtKernel_Type`だけである。

```fortran
NeX = 1; NeY = 1; NeZ = 1
PolyOrder = 1023
dt = 1.0D-8
nstep = 15
output_interval = 15
WarmupStep = 3
UseCudaGraph = .false.
MeasureKernelTime = .true.
CublasEmulation = .false.
CutlassMmaShape = "8x8x4"
```

各経路を3回ずつ実行した。各runでは先頭3 stepを実行するが計時から除き、残り
12 step = 36 RK stagesを集計した。表は3 runの中央値である。巨大host/device
allocationと初期copyはMainおよびdevice eventの計測区間外である。

### 6.2 結果

| 経路 | Main [s] | Main [ms/step] | device tendency [s] | device [ms/stage] |
|---|---:|---:|---:|---:|
| `CUDAFORTRAN_GEMM` | 6.94625 | 578.854 | 6.98610 | 194.058 |
| `CUDAFORTRAN_GEMM_FUSED` | 6.75001 | 562.501 | 6.75420 | 187.617 |

GEMM_FUSEDはdevice tendencyで3.32%、end-to-end Main/stepで2.83%短い。
GEMM_FUSED内のvolume GEMM区間は中央値6.44566 s、179.046 ms/stageだった。

各runのdevice tendency合計はGEMMが6.95583 / 7.10037 / 6.98610 s、
GEMM_FUSEDが6.76640 / 6.75420 / 6.75351 sである。

この測定では`ncu`を使っていないため、profiler measured FLOP/sやbandwidthは
報告しない。理論volume workは3方向合計
`6*Nq^4 = 6,597,069,766,656` FLOP/stageだが、device tendencyにはpointwise
flux、全6面のnumerical flux、lift、assemblyも含まれる。

## 7. 結論

- p=1023はGB200 189471 MiB上でGEMM/GEMM_FUSEDとも実行可能になった。
- 64-bit化は境界を越えるbase pointer形成に限定し、point indexは32-bitのままにした。
- 両経路の主要device配列は144 GiBである。
- 点変化係数を含む全2^30点の`dqdt`は丸め誤差範囲で一致した。
- p=1023でも最速は`CUDAFORTRAN_GEMM_FUSED`で、device時間の優位は3.32%だった。

## 8. Ozaki Scheme I / II（2026-08-28 追記）

p=511 以降のゲート拡張後、同一 DOF（`Ne=1`）で Ozaki 経路を試した。
commit `38952e4`、`nstep=20`、`WarmupStep=2`、
`OzakiSliceCount=8` / `OzakiModuliCount=14`、GB200 login ノード。

| 経路 | µs/stage | 結果 |
|---|---:|---|
| `CUDAFORTRAN_GEMM` | **575.5 ms** | 成功 |
| `CUDAFORTRAN_GEMM_OZAKI1` | — | **OOM**（`cuMemAlloc`、workspace 確保時） |
| `CUDAFORTRAN_GEMM_OZAKI2` | — | **OOM**（同上） |

p=767 では Scheme I が native を上回ったが、p=1023 では Ozaki workspace
（`iB` / `scale_a` / OZAKI2 residue 等）が **144 GiB payload に上乗せ**され、
189 GiB GPU で setup 中に OOM となる。`OzakiSliceCount` / `OzakiModuliCount`
の削減、または workspace 分割は未実施。

> **（訂正 2026-08-29）** 上表の native **575.5 ms** は 1 step の device 合計で、
> RK 3 stage で割っていない。1 stage は約 192 ms で、§6 の GEMM 194.058 ms/stage
> と整合する。OOM の結論は変わらない。

## 9. cuBLAS FP64 emulation（2026-08-28 追記）

`CublasEmulation = .true.`、`CUDAFORTRAN_GEMM` のみ。commit `a1cdb57`、
§6 と同条件（`nstep=15`、`WarmupStep=3`）、GB200 login ノード。

| 経路 | device [ms/stage] | native GEMM 比 | 備考 |
|---|---:|---:|---|
| `CUDAFORTRAN_GEMM`（native） | 194.06 | 1.00 | §6 |
| `CUDAFORTRAN_GEMM` + emulation | **338.2** | **1.74×** | 2 run 中央値 |

3 run 目は 144 GiB payload 確保後、emulation 用 8 GiB workspace 確保で OOM
（free ≈ 6.2 GiB）。2 run は 268.7 / 407.6 ms/stage と run 間変動が大きい。
入力は `input_p1023_val_gemm_emu.conf`。
[`cublas_emulation_survey.md`](cublas_emulation_survey.md) §4 参照。

## 10. なぜ emulation は p=1023 でも native GEMM より遅いか（2026-08-29）

孤立 GEMM ベンチ（`bench_ozaki2/p1023_emu_why.cu`、GB200、176 GiB の場は無し）で
§9 の 1.74× を分解した。詳細表は
[`cublas_emulation_survey.md`](cublas_emulation_survey.md) §6。

- volume GEMM 3 本の native 合計は **170.0 ms**（各方向とも FP64 ピークの 96%）。
  `CublasEmulation=.true.`（EAGER + ADP）は **385.3 ms（2.27×）**。
  残り ~24 ms の flux/面/assembly は対象外なので、tendency 予測は
  409 ms/stage。§9 の遅い run（407.6 ms/stage）と一致する。
- ADP は 54 mantissa bit を選ぶ。同じ 55 bit を FIXED にすると x は
  **0.495×** まで速くなる。負けの本体はスライス数ではなく
  **DYNAMIC が毎呼び出しで 8 GiB の B を解析するコスト**。
  同日、本体の `CublasEmulation=.true.` はこの FIXED 55 に切り替えた
  （[`cublas_emulation_survey.md`](cublas_emulation_survey.md) §7）。
- y だけは FIXED 55 でも **1.41× 遅い**（1024 本の 1024³。native が既に
  ピーク 97% で、INT8 タイルが埋まらない）。x/z の勝ちを食う。
- INT8 1 本の x 形状は 2464 TOP/s（native の 64 倍）なので、Scheme II なら
  算術下限で勝つ。cuBLAS 13.1 は Scheme I。`OZAKI2` は workspace が payload に
  乗って OOM（§8）。

## 11. 2026-08-29 の経路横断について

p=7…255 は同一実行ファイルで再測定したが、**当時 p=1023 は再実行していない**。
[`README.md`](README.md) まとめ表の p=1023 行は本レポート §6 の
`GEMM` 194.058 / `GEMM_FUSED` 187.617 ms/stage のまま。
§12 の login 再測（185.851 ms/stage）は表を書き換えない。

## 追記（2026-09-01）: p=575 側から入った `Nq >= 512` の分岐

[`p575_gap_study.md`](p575_gap_study.md) §11 が volume GEMM に `Nq >= 512` の
分岐を 2 つ足したので、本次数にもそのまま載る。
本レポート本文の 187.617 ms/stage は 2026-08-28 の値で、`p511_gap_study.md` §12
の融合 y の 3 段化（本次数で −0.06%）より前である。

- **§11.2**: 融合 x を `64x128` の 1 本から、y と同じ `64x64` batched へ。
- **§11.13**: その x も `GemmYScaleShallow` を共有して 3 段パイプラインにする
  （`GEMM_CUTE` の x / y も同じ mainloop）。

両方入った実行ファイルと、batched x だけ入れて x/y は 4 段のままの実行ファイルを
占有 GPU 上で 8 回交互に測った（`p575_gap_study.md` §11.16、job `74975`、
`namelists/perf_p1023_gemm_fused.conf`）:
**189027.1 → 188694.3 µs/stage（−0.176%）**。この A/B が測っているのは
**batched x の上で x/y を 4 段から 3 段にする効果**であって、
`p511_gap_study.md` §12 が融合 y 単独で測った値とは分母も対象も違う。
数値は p=575 で 191,102,976 点ビット一致（§11.13）。


## 12. `GEMM_FUSED` の残り天井（2026-08-31）

対象は `CUDAFORTRAN_GEMM_FUSED`、`namelists/perf_p1023_gemm_fused.conf`
（`Ne=1`、`nstep=15`、`WarmupStep=3`、graph off）。commit `f7788bb`、
login GPU 2 で 3-run 中央値。ncu はクロック固定なので採否に使わず、
機構の同定だけに使う。

### 12.1 ベースライン

| 量 | run1 | run2 | run3 | 中央値 |
|---|---:|---:|---:|---:|
| Main [ms/step] | 557.349 | 557.368 | 557.121 | **557.349** |
| device GEMM fused [s / 36 stage] | 6.69064 | 6.69084 | 6.68818 | **6.69064** |
| device [ms/stage] | 185.851 | 185.857 | 185.783 | **185.851** |
| volume GEMM only [ms/stage] | 177.302 | 177.305 | 177.229 | **177.302** |

§6 の 187.617 ms/stage は別セッション。表は書き換えない。
6.619e12 FLOP / 185.851 ms = **35.61 TFLOP/s（ピーク 40.1 の 88.8%）**。
unique DRAM 68971 MB では 0.371 TB/s（7.9 の 4.7%）。

### 12.2 nsys（job `71135`、c178）

`nstep=4`、12 RK stage。カーネル中央値:

| カーネル | med [ms] | 孤立 cuBLAS（§10） | 差 |
|---|---:|---:|---:|
| x `Gemm` + Escale epilogue | 56.860 | 56.969 | **−0.11** |
| y `GemmBatchedScaleAdd` | 56.380 | 56.521 | **−0.14** |
| z `GemmBatchedDqdtAssembly` | 58.613 | 56.491 | **+2.12** |
| `volume_flux_kernel` | 8.312 | — | — |
| `elembnd_flux_kernel` | 0.314 | — | side stream |

x/y は孤立 cuBLAS（ピーク 96–97%）と互角。タイル掃引と x の cuBLAS 差し替えは
賞金が測定誤差以下。z の +2.12 ms が融合 epilogue の税。`volume_flux` は
4 読 3 書 = 56 GiB、7.9 TB/s 屋根なら 7.09 ms。

### 12.3 ncu（job `73612` flux、`73617` x/y/z、c179、`--set full`）

ncu 時間はクロック固定で壁時間の約 2 倍。機構だけ読む。

| カーネル | SM | DRAM | 占有率 | 命令数 | 律速 |
|---|---:|---:|---:|---:|---|
| `volume_flux` | 44.5% | **91.25%**（7.23 TB/s） | 80.5% | 1.17e9 | **DRAM 91%** |
| x GEMM | **97.36%** | 10.2% | 12.5%（reg=shared=2 CTA） | 14.67e9 | **SM / tensor pipe** |
| y scaleadd | **97.41%** | 4.1% | 18.8% | 17.52e9 | 同左 |
| z assembly | **94.90%** | 4.2% | 12.5%（reg=shared=4 CTA、254 レジスタ） | 21.56e9 | **SM 95%、発行スロット 29%** |

**x/y はこのカーネルは SM が 97% で律速している。** z は同じ屋根の 2.5 ポイント下。
z の命令は x より 47% 多いが時間は 3.6% しか伸びない。p=255 §10.2 の
「lift を消すと命令 −13% で時間 −12.8%」は、ここでは成り立たない
（発行スロットは 29% しか埋まっていない）。

`volume_flux` は **DRAM 91% で律速している。** 契約が要求する 56 GiB を
動かしている限り、ベクトル化も TMA もバイトを減らさない。屋根 100% までの
天井は 8.31×(1−0.9125) ≈ **0.73 ms（stage の 0.39%）**。

### 12.4 lift を消したアブレーション（不正。login GPU 2、3-run）

z epilogue の面 gather と積和を落とし、`dqdt = -(dxy + Ez*Dz)` だけにした。
数値は壊れる。device GEMM fused 中央値 **6.64310 s = 184.531 ms/stage**
（6.64261 / 6.64310 / 6.65165）。基準 185.851 に対し **−1.32 ms（−0.71%）**。
孤立 z との 2.12 ms 差の残りは Ez と `dqdt` ストアとアキュムレータの shared 往復で、
いずれも出力に必要。残る lift は 1 要素 3 本の 16 バイトロードという
p=255 §10.6 の下限のまま。**契約内で取る手は無い。**

コードは測定後に戻した。

### 12.5 CUDA graph（login GPU 3、3-run）

同一 conf で `UseCudaGraph = .true.`。Main 555.472 / 555.404 / 555.382 ms/step、
中央値 **555.404**。graph off の 557.349 に **−0.35%**（レンジ非重複）。
device 時間は graph では非計測。既定 conf は off のまま（`nsys` が graph を
取れないことと、§6 以降の比較を固定するため）。

### 12.6 測って閉じた候補

| 候補 | 天井 | 判定 |
|---|---|---|
| x/y を cuBLAS に戻す / タイル掃引 | x/y が孤立 cuBLAS より速い | 実装せず |
| z の命令削減（p=255 型） | lift 全消しでも −0.71% | 契約内ゼロ。戻した |
| `volume_flux` のベクトル化 / TMA / CTA | DRAM 91%、バイトは契約 | 屋根 0.39%。実装せず |
| flux を x GEMM に重ねる | p=63 は +5.9%。x は flux_x を読む | 順序が成立しない |
| z `launch_bounds` で占有率 | p=63 は +90%（spill）。254 レジスタ | 実装せず |
| RK を z epilogue に融合 | `dqdt` を実体化しない | 範囲外 |
| Ozaki / emulation | §8–§10 | OOM または 1.74× |

### 12.7 結論

カーネル変更の採用はゼロ。最速は `CUDAFORTRAN_GEMM_FUSED` のまま
（login 再測 **185.851 ms/stage**、§6 の 187.617 は当時値）。
x/y は FP64 ピークの 97%、z は 95%、flux は DRAM 91%。契約内に残る天井は
lift 全消しの 0.71% と flux 屋根の 0.39% で、どちらも実装できないか
バイトそのもの。**p=1023 `GEMM_FUSED` の探索はここで終了する。**

> **訂正（2026-09-01、§13）**: §12.2 の内訳は stage 合計 185.6 ms と 3 GEMM の
> 和 171.9 ms の差 13.7 ms を説明していなかった（正体は `rk_update` 4.79 ms と
> `volume_flux` 8.31 ms で、ローンチの隙間は 0.03 ms）。また §12.6 が
> 「順序が成立しない」として机上で落とした flux の重ねは、§13.3 で天井を
> 実測して 0.10% と確定した。§12.6 が掃いた tile は x/y だけで、z は
> §13.4 で掃いた。表の数値は当時値のまま書き換えない。

## 13. `GEMM_FUSED` の stage 全内訳と、閉じ直した候補（2026-09-01）

§12 は「探索終了」で閉じたが、その後 `feature/cuda` に
`run_gemm_batched_nn_capped` の単一ローンチ化（`2e48571`）と
`RepadEpilogue<...,8>`（`09cb3b3`）が入った。どちらも p=1023 が通る経路には
届いていないので、ベースラインから測り直し、§12 が**机上で**閉じた候補を
**天井の実測**で閉じ直した。

対象は `CUDAFORTRAN_GEMM_FUSED`、`namelists/perf_p1023_gemm_fused.conf`
（`Ne=1`、`nstep=15`、`WarmupStep=3`、graph off、`CutlassMmaShape = "8x8x4"`）。
commit は本節を追加したもの（親 `acdbd8a`）。

> **基準について（2026-09-01 追記）**: 本節の測定はすべて `nstep=15` 当時の
> conf（12 measured step = 36 stage）で、per-stage は
> `CUDA device GEMM fused ÷ 36`。その後 `feature/cuda` で conf が `nstep=10` に
> なり、`p767_gap_study.md` §15 がこれらのタイマに **1 stage 分の偏り**が
> あることを示した（イベント系は `÷(3·steps+1)`）。**A/B の差分はどれも同一
> conf・同一分母なので無傷**だが、絶対値は旧基準である。なお p=1023 では
> `÷36 = 185.60 ms/stage` が §13.2 の trace 実測の stage 周期 186.06 ms と
> 0.25% で一致し、`÷37 = 180.58` は 3% ずれる。p=767 で導いた分母がこの次数の
> trace と合わない点は未解決で、絶対値を次数横断で並べるときは要確認。

### 13.1 ベースライン再測（login GPU 2、3-run）

| 量 | run1 | run2 | run3 | 中央値 |
|---|---:|---:|---:|---:|
| Main [ms/step] | 556.954 | 556.583 | 556.273 | **556.583** |
| device GEMM fused [s / 36 stage] | 6.68580 | 6.68149 | 6.67780 | **6.68149** |
| device [ms/stage] | 185.717 | 185.597 | 185.494 | **185.597** |
| volume GEMM only [ms/stage] | 177.164 | 177.048 | 176.941 | **177.048** |

§12.1 の 185.851 と差は −0.14%（レンジ重複）。**共有コードの 2 変更は p=1023 に
届いていない**: capped launcher は batch ≤ 65535 の p=1023 には無関係で、
repad が入ったのは `run_gemm_batched_nn_capped`（`GEMM_CUTE` の y/z と
`Nq<64` 融合 y）であり、`Nq>=64` の融合 y は別ローンチャだからである。

### 13.2 stage の全内訳（nsys job `74738`、c183、`nstep=4`）

§12.2 はカーネル中央値だけを並べ、stage 合計 185.6 と 3 GEMM の和 171.9 の差
13.7 ms を説明していなかった。GPU trace の開始・終了時刻から stage 1 周期
（`volume_flux` の開始から次の `volume_flux` の開始まで）を分解する:

| 要素 | stream | ms/stage | stage 比 | 律速 |
|---|---|---:|---:|---|
| `volume_flux` | main | 8.311 | 4.47% | DRAM 91%（§12.3） |
| x GEMM + Escale epilogue | main | 57.14 | 30.7% | FP64 ピークの **96.0%** |
| y `GemmBatchedScaleAdd` | main | 56.68 | 30.5% | **96.8%** |
| z `GemmBatchedDqdtAssembly` | main | 58.86 | 31.6% | **93.2%** |
| `rk_update_stage`（RK 時間積分） | main | 4.79 | 2.58% | DRAM **90.8%** |
| halo 更新 | main | 0.075 | 0.04% | — |
| `elembnd_flux` | side | 0.32 | — | 完全に隠れている |
| **カーネル間の隙間の合計** | — | **0.03** | **0.02%** | — |
| stage 周期 | | **186.06** | | |

- **説明できていなかった 13.7 ms の正体は `rk_update` 4.79 ms と
  `volume_flux` 8.31 ms**であり、ローンチの隙間ではない。3 本の GEMM は
  隙間 7 µs で連続している。**ローンチ overhead を削る候補は天井 0.02% で
  閉じる**（§12.5 の CUDA graph −0.35% は、隙間ではなく CPU 側の
  ローンチ列の話である）。
- 3 GEMM 合計 172.68 ms、6.597e12 FLOP で **38.20 TFLOP/s = ピーク 40.1 の
  95.3%**。
- `rk_update` は `main.f90:401`。`qq0`, `qq`, `dq` を読み `qq` を書く 4 pass ×
  8.59 GB = 34.36 GB を 4.787 ms、**7.18 TB/s = 7.9 の 90.8%**。
  `volume_flux` と同じ DRAM 屋根で、しかも DG カーネルの外である。

### 13.3 `volume_flux` を GEMM に重ねる天井（不正アブレーション、login GPU 2、3-run）

§12.6 はこの候補を「x が `flux_x` を読むので順序が成立しない」と**机上で**
落としていた。順序を無視して天井だけを測る。`volume_flux` を side2 stream に
出し、x/y GEMM の前で待たない（x/y は前 stage の flux を読むので数値は壊れる。
1-step min/max が `-9.999992752565223E-01` と、正しい `…5273E-01` から
ずれることで確認）。join は z assembly の直前。

| 量 | 基準 | flux 重ね | 差 |
|---|---:|---:|---:|
| Main [ms/step] | 556.583 | 556.016 | **−0.10%** |
| device [s / 36 stage] | 6.68149 | 6.67522 | **−0.094%** |

**`volume_flux` を丸ごと隠しても 0.10% しか返ってこない。** stage の 4.47% を
占める 8.31 ms に対し、回収できたのは 0.19 ms/stage である。x/y GEMM が SM を
96–97% 占めているので、DRAM 律速の `volume_flux` は同時実行しても発行スロットを
取れない（§12.3 で `volume_flux` 自身が占有率 80.5% を使って DRAM 91% に
達していることの裏返し）。したがって
**「`flux_x` 専用カーネルを先に出して `flux_yz` を x GEMM に重ねる」分割版
（読み書き 7 pass → 8 pass、理想で −2.6% と見積もっていた）は実装しない。**
コードは測定後に戻した。

### 13.4 z assembly の tile 掃引（login GPU 2、3-run）

§12.6 が閉じたのは x/y の tile であり、z は掃いていなかった。z だけが
ピークの 93.2% で、しかも既定の 64x32 / warp 32x32 は **2 warp（64 スレッド）
/CTA** しか使わない（x は 64x128、y は 64x64 で 4 warp）。`Nq>64` 用の
order-specialized set を `GEMM_CUTE` と `GEMM_FUSED` の両方に入れて掃いた。

| z tile / warp / stage | device [s / 36 stage] | 基準比 |
|---|---:|---:|
| **64x32 / 32x32 / 4（既定）** | **6.68149** | — |
| 64x32 / 32x32 / 3 | 6.68242 | +0.01%（差なし） |
| 64x64 / 32x32 / 4 | 6.70799 | +0.40% |
| 128x32 / 32x32 / 4 | 6.71115 | +0.44% |
| 64x32 / 32x16 / 4 | 6.72461 | +0.65% |

いずれも 1-step min/max は既定と一致（数値は不変）。**既定の 64x32 が最速で、
warp を増やす方向も N を広げる方向も負け。** N を広げると CTA あたりの
shared が増えて SM あたりの CTA が減り、z が頼っている latency hiding が
落ちる。z の残りは §12.4 が測ったとおり lift 全消しでも 0.71% で、tile では
取れない。コードは測定後に戻した。

### 13.4b z epilogue の pad 値と 16 B アクセス（占有 GPU job `74953`、10 回交互）

§13.4 は tile を掃いた。epilogue 側に残る 2 つのつまみも決着させる。

| 変種 | device 中央値 [s / 36 stage] | pad=8 比 | 判定 |
|---|---:|---:|---|
| **`RepadEpilogue` pad=8（既定）** | **6.662160** | — | 最速 |
| pad=4 | 6.663475 | +0.020% | 100 組中 97 組で既定が速い |
| pad=16 | 6.664080 | +0.029% | **レンジ非重複**で既定が速い |

16 B epilogue（`GemmZWide` = `EpilogueOp2`）を切って `EpilogueOp` に戻す版は
login 3-run で 6.68610 s（最小 6.68126）で、既定 6.67693 の中央値を最小値でも
上回った。**`Nq>64` で 16 B アクセスを使う既定は正しい**（`Nq=64` で
+2.8% だったのと逆であることを確認した）。数値はどの変種でも 1-step min/max
が一致する。

### 13.5 融合 y epilogue の repad（採用、占有 GPU job `74733`、c182、12 回交互）

`09cb3b3` が `RepadEpilogue<...,8>` を入れたのは `run_gemm_batched_nn_capped`
だけで、`Nq>=64` の融合 y（`run_volume_gemm_y_scaleadd` の
`GemmBatchedScaleAdd`）は stock の epilogue のまま、つまり accumulator の
shared staging が無 padding のままだった。z assembly と同じ pad を入れる。

> **注（2026-09-01）**: 同じ欠落を `p767_gap_study.md` §12 が独立に見つけ、
> **x（`run_gemm_nn_scaled`）と y の両方**に pad を入れて先に採用した
> （`fc272e9`。p=767 −0.127%、p=127 −0.487%、ncu で shared store conflict
> −87%、命令数は不変）。本節が測っているのは**その y 単独分**であり、
> 実装は `fc272e9` に置き換わっている。

| 量 | base 中央値 | repad 中央値 | 差 | レンジ |
|---|---:|---:|---:|---|
| device GEMM fused [s / 36 stage] | 6.665640 | 6.664345 | **−0.019%** | 非重複 |
| volume GEMM only [s] | 6.357690 | 6.356425 | −0.020% | 非重複 |
| Main [ms/step] | 555.311 | 555.201 | −0.020% | **重複** |

**再現するが 0.02% である。** p=1023 の y は K=1024 で、epilogue 1 回に対し
mainloop が 64 イテレーション回るので、pad が消す shared store conflict は
mainloop に対して割に合わない（p=7 で −4.11% だったのは K=8 で epilogue の
比率が 8 倍大きいため）。採用理由は速度ではなく、
**3 つの volume GEMM epilogue の staging を揃えること**であり、
出力は 1-step min/max がビット一致する。

数値検証: 点変化係数（`SCALE_DG_VARYING_COEFF=1`）の 1-step `dqdt` を
全 1,073,741,824 点ダンプし、repad 前後で **`cmp` がビット一致**
（26,843,545,600 byte）。pad は shared のレイアウトしか変えないので当然だが、
§5 の全点検証を再走せずに済ませない形で確認した。加えて
`CUDAFORTRAN_SPLIT` との全場比較を p=63 で 3.55e-15、p=127 で 1.78e-15。

### 13.6 横展開（占有 GPU job `74846`、c182、12 回交互）

repad が入る `run_volume_gemm_y_scaleadd` は `Nq>=64` の融合 y 共通なので、
同じ実行ファイルで他の次数も測った。

| 次数 | 経路 | base 中央値 [s] | repad 中央値 [s] | 差 | 判定 |
|---|---|---:|---:|---:|---|
| p=63（`Nq=64`、`kMulAddend=true`） | GEMM_FUSED | 0.032521 | 0.032512 | −0.027% | **差なし**（144 組中 92 組） |
| p=127（`Nq=128`） | GEMM_FUSED | 0.192294 | 0.191706 | **−0.306%** | 有意（144 組中 **143** 組で repad が速い） |
| p=1023（`Nq=1024`） | GEMM_FUSED | 6.665640 | 6.664345 | −0.019% | 採用（レンジ非重複） |

K が深いほど epilogue の比率が下がるので、効果は Nq とともに単調に小さくなる
（p=7 −4.11% → p=15 −0.63% → p=127 −0.31% → p=1023 −0.019%、p=63 は
`kMulAddend` 版で差なし）。p=127 の最速は `FUSED_TC`（621.5 µs/stage）のままで、
この変更は `GEMM_FUSED` 側だけを動かす。ここの p=127 −0.306% は **y 単独**で、
`p767_gap_study.md` §12 が x と合わせて測った −0.487%（job `74821`）の内訳に
あたる。

### 13.8 リベース後の現在地（占有 GPU job `75692`、c-node、12 回交互）

§13 の測定後に `feature/cuda` が 7 コミット進み、うち `fc272e9`（融合 x/y
epilogue の repad）、`b344c6d` / `04172b0`（`Nq>=512` の batched x と x/y の
3 段パイプライン）が本次数の経路に載る。リベース前後の実行ファイルを
**同一 conf**（現行 `nstep=10`、`WarmupStep=3` = 21 stage）で交互に測った。

| 量 | リベース前 | リベース後 | 差 |
|---|---:|---:|---:|
| Main [ms/step] | 555.947 | 555.674 | **−0.049%** |
| `Cal_tend` [s] | 3.697680 | 3.695915 | −0.048% |
| device GEMM fused [s] | 3.966235 | 3.964310 | −0.049% |
| volume GEMM only [s] | 3.783110 | 3.781175 | −0.051% |

**4 つのタイマすべてで 144 組中 144 組・レンジ非重複**。1-step の min/max は
両者で完全に一致する（`-9.999992750387978E-01` / `9.999992759126788E-01`）。
上流の 3 変更を合わせて **−0.049%** で、内訳としては §13.5 が測った y 単独分
（−0.019%）に x の repad が乗った程度である。`Nq>=512` の batched x と
3 段パイプラインは、この次数では単独で測れる差を足していない（本レポート
「追記」の job `74975` が別の対で −0.176% としているのと分母も対も違う）。

分母について: 上流の「追記」が引く 188694.3 µs/stage は
`device GEMM fused ÷ 21 = ÷(3·steps)` であり、本節の 3.964310 s も
÷21 で 188.78 ms/stage になる。つまり**この次数では上流自身も
`÷(3·steps+1)` を使っていない**。§13 冒頭の注記のとおり、分母の扱いは
p=767 §15 と未整合のままである。

### 13.7 結論

本節で新たに採用したカーネル変更はゼロである。融合 y epilogue の repad は
測定した唯一の正の効果（p=1023 **−0.019%**、p=127 −0.31%）だが、実装は
`p767_gap_study.md` §12 が x と一緒に入れた `fc272e9` に置き換わった。
p=1023 `GEMM_FUSED` は旧基準で **185.56 ms/stage**（占有 GPU 換算 185.12）。

§12 が机上または部分的に閉じていた候補を、天井の実測で閉じ直した:

| 候補 | 天井（実測） | 判定 |
|---|---|---|
| `volume_flux` を GEMM に重ねる（分割込み） | **0.10%**（丸ごと重ねて） | 実装せず。§12.6 の「順序が成立しない」を測定で置き換えた |
| ローンチ overhead / カーネル間の隙間 | **0.02%**（trace 実測 0.03 ms） | 閉じた |
| z assembly の tile | 4 形すべて既定以上（+0.01〜+0.65%） | 既定が最速 |
| z epilogue の pad 値 / 16 B アクセス | pad 4 は +0.020%、16 は +0.029%、非 wide は負け | 既定が最速（§13.4b） |
| z の命令削減 | §12.4 の lift 全消しで −0.71% | 契約内ゼロ |
| `volume_flux` のバイト削減 | DRAM 91%、屋根まで 0.39% | バイトは契約 |
| `rk_update` | DRAM 90.8%、かつ DG カーネル外 | 範囲外 |

stage 周期 186.06 ms のうち **92.8%（172.68 ms）は FP64 ピークの 95.3% で回る
3 本の GEMM**、**7.0%（13.10 ms）は DRAM 屋根の 91% で回る `volume_flux` と
`rk_update`**、残り 0.2% が halo とカーネル間の隙間である。
**p=1023 `GEMM_FUSED` は契約内で機械の屋根に張り付いており、探索はここで
終了する。**
