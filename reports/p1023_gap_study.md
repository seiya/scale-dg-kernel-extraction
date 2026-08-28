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
