# cuBLAS FP64 fixed-point emulation のストール調査

- 日付: 2026-08-27
- 対象: base commit `1d9b7ec` 上の本変更
- GPU: NVIDIA GB200、compute capability 10.0
- build: NVIDIA HPC SDK 26.3、`make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100`
- cuBLAS runtime: 13.2.1
- timing jobs: Slurm `53849`（nativeとp=255、host `c390`）、`54394`
  （追跡対象p=7 emulation入力の確認、host `c188`）
- validation job: Slurm `53850`、host `c392`
- nsys job: Slurm `53863`、host `c392`
- profiler command: `timeout 180 nsys profile --trace=cuda --sample=none --cpuctxsw=none --resolve-symbols=false --force-overwrite=true -o output/cublas_emu_p7_final ./scale-dg_extraction.emuworkspace input_p7_perf_gemm_emu.conf`

## 1. 結論

観測された停止は deadlock ではない。`CublasEmulation=.true.` が cuBLAS の
`EAGER` strategy を強制し、p=7 の K=8 GEMM にも FP64 fixed-point emulation を
適用した結果、入力の範囲解析、INT8 slice への pack、複数の低精度 GEMM、FP64
への再構成が本来の小さな GEMM を圧倒していた。

このフラグの目的は計算方法の比較なので、自動選択の `PERFORMANT` には変更しない。
emulation on は今後も `EAGER`、off は native FP64 とする。

## 2. 修正した実装上の誤り

旧コードは `CUBLAS_EMULATION_STRATEGY_EAGER` という enum値を
`#if defined(...)` で検査していた。enum値はpreprocessor macroではないため分岐は
常に偽になり、「API unavailable; native FP64 GEMM will be used」と表示された。
しかし、その前に `CUBLAS_EMULATE_DOUBLE_PRECISION=1` と
`CUBLAS_EMULATION_STRATEGY=eager` を設定していたので、実際にはemulationが動いた。

判定を `CUBLAS_VERSION >= 130002` に変更した。古いcuBLASではnativeへ黙って
fallbackせずエラーにするため、比較対象を取り違えない。

API分岐が有効になると、旧コードのemulation off側にあった
`CUBLAS_PEDANTIC_MATH` も初めて作用し、native側だけを遅くすることが分かった。
比較条件を元のnative DGEMMに保つため、off側は `CUBLAS_DEFAULT_MATH` に戻した。

## 3. 永続workspace

cuBLASが文書化するfixed-point workspaceの最大値8 GiBを、emulation有効時の
初期化で一度だけ `cudaMalloc` する。OpenACC queueのCUDA streamを先に取得し、
cuBLAS handle作成後に `cublasSetStream`、続いて `cublasSetWorkspace` を一度だけ
実行する。全RK stageと全GEMMが同じpointerを再利用し、終了時に解放する。

ただし、cuBLAS 13.2.1はこのworkspaceとは別の内部 `cudaMallocAsync` を残す。
p=7、1 stepのnsysでは、workspace導入前（job `53776`）の27回から導入後
（job `53863`）の15回には減ったがゼロにはならなかった。後者では
`cudaMallocAsync` が合計682.8 msで、依然として最大のhost-side costだった。
したがって、永続workspaceは実装したが、p=7の低速化を解消するものではない。

## 4. 速度

nativeは通常のperformance入力を1000 step、emulationは長時間化を避けるため
1 stepを3回測った。表のemulationは3回の中央値。CUDA device timeは1 step
（SSP-RK3の3 tendency）の値である。

| order / method | 入力 | nstep | CUDA device / step | native比 |
|---|---|---:|---:|---:|
| p=7 native | `bench_runs/p7_gemm.conf` | 1000 | 5.111 ms | 1.00 |
| p=7 EAGER emulation | `input_p7_perf_gemm_emu.conf` | 1 | 667.642 ms | **130.6倍遅い** |
| p=255 native | `bench_runs/p255_gemm.conf` | 1000 | 3.496 ms | 1.00 |
| p=255 EAGER emulation | `input_p255_val_gemm_emu.conf` | 1 | 30.748 ms | **8.80倍遅い** |
| p=511 native | §5 参照 | 20 | 40.58 ms/step | 1.00 |
| p=511 EAGER emulation | `input_p511_val_gemm_emu.conf` | 20 | 60.68 ms/step | **1.50倍遅い** |
| p=575 native | §5 参照 | 50 | 62.74 ms/step | 1.00 |
| p=575 EAGER emulation | `input_p575_val_gemm_emu.conf` | 50 | 131.9 ms/step | **2.10倍遅い** |
| p=767 native | §5 参照 | 30 | 188.5 ms/step | 1.00 |
| p=767 EAGER emulation | `input_p767_val_gemm_emu.conf` | 30 | 366.1 ms/step | **1.94倍遅い** |
| p=1023 native | §6 参照 | 15 | 582.2 ms/step | 1.00 |
| p=1023 EAGER emulation | `input_p1023_val_gemm_emu.conf` | 15 | 1014.6 ms/step（2 run 中央値） | **1.74倍遅い** |

p=7 emulationの3回は634.964 / 667.642 / 1058.470 msで、内部allocation由来の
run間変動も大きい。表は中央値を使った。

p=511 以降も emulation は3 run 中央値（p=1023 は3 run 目が 8 GiB workspace
確保時 OOM のため2 run 中央値）。表の device 時間は 1 time step あたり
（SSP-RK3 の 3 tendency 合計）。p=1023 の2 run は 806 / 1223 ms/step と
変動が大きい（p=7 と同様、cuBLAS 内部 allocation の影響）。

p=7の通常計測で使う `nstep=1000` をそのままemulationへ適用してはいけない。
初回の機能・速度確認は **`nstep=1--10`** とする。p=255も比較の初回は
`nstep=1--100` に抑え、1 step時間を確認してから延ばす。

## 5. 数値検証

`SCALE_DG_VARYING_COEFF=1` と `SCALE_DG_DUMP_DQDT` を使い、native FP64と
EAGER emulationの全owned `dqdt`を比較した。

| order | owned点数 | maxabs | 参照maxabs |
|---|---:|---:|---:|
| p=7、Ne=2x2x2 | 4,096 | 0 | 8.1808 |
| p=255、Ne=1 | 16,777,216 | 7.461e-14 | 8.5482 |

p=7はビット一致、p=255はFP64 roundoffレベルで一致した。入力は
`input_p7_val_gemm.conf` / `input_p7_val_gemm_emu.conf` と
`input_p255_val_gemm.conf` / `input_p255_val_gemm_emu.conf`。

## 6. なぜ p=1023 でも EAGER は native DGEMM より遅いか（2026-08-29）

- 日付: 2026-08-29
- GPU: NVIDIA GB200、index 1、189471 MiB
- build: NVIDIA HPC SDK 26.3、`nvcc -O3 -arch=sm_100`、CUDA 13.1 / cuBLAS 同梱
- 測定: `bench_ozaki2/p1023_emu_why.cu`（volume GEMM 3 本だけ。176 GiB の DG 場は載せない）
- native 対照: 同一バイナリを `CUBLAS_EMULATE_DOUBLE_PRECISION=0` で起動
- emulation: **プロセス起動前**に `CUBLAS_EMULATE_DOUBLE_PRECISION=1` と
  `CUBLAS_EMULATION_STRATEGY=eager` を export。`cublasCreate` のあとに
  `setenv` してもライブラリは読まない（同一プロセスで native → emu と切り替えると
  `bits_used=-1` のまま時間が native と 0.01% 以内で一致する）

§4 の p=1023 比 1.74× は 2 run（806 / 1223 ms/step）の中央値で、3 run 目は
8 GiB workspace 確保時 OOM。本節は GEMM 単体でその比の中身を分解する。

### 6.1 結論

p=1023 で負けているのは「K がまだ浅いから INT8 が足りない」ではない。
x 形状の INT8 1 本は **2464 TOP/s（native FP64 38.60 TFLOP/s の 63.8 倍）**で、
7 スライスの Scheme I 算術下限は native の 0.77 倍まで下がる。負けの本体は 2 つ。

1. **測定当時の `CublasEmulation=.true.` は ADP（DYNAMIC mantissa）だった。**
   （2026-08-29 以降は FIXED 55。§7。）
   Chebyshev D（LGL `D1D` の同型、`log2(max/min)=29`）に対し ADP は **54 bit**
   を選ぶ。同じ 55 bit を FIXED で渡した x GEMM は native の **0.495 倍**なのに、
   ADP 付き EAGER は **2.00 倍遅い**。bit 数ではなく、入力解析そのものが
   8 GiB の B を毎呼び出し走査するコストである。
2. **y GEMM は 1024 本の 1024³ batched で、FIXED 55 でも native より 1.41 倍遅い。**
   native y は既に 38.91 TFLOP/s（ピーク 40.1 の 97%）。1024³ は INT8 タイルが
   埋まらず、x で出た 2464 TOP/s はここでは出ない。NVIDIA の heatmap が勝つ
   「大きな正方（8192³）」とは形が違う。

Ozaki Scheme II（s 本、CRT）なら算術だけでも 14×0.89 ms = 12.5 ms 対 native
57 ms で、[`ozaki2_survey_2504.08009.md`](ozaki2_survey_2504.08009.md) §6.1 の
p=1023 予測 0.58× と向きは同じ。cuBLAS 13.1 は Scheme I（ADP / スライス直積）
であり、NVIDIA 自身も Ozaki-II は将来版としている。本体の
`CUDAFORTRAN_GEMM_OZAKI2` は 144 GiB payload の上に residue を足して p=1023 では
OOM（`p1023_gap_study.md` §8）。

### 6.2 p=1023 volume GEMM 3 本（Ne=1, Nq=1024）

| 呼び出し | 形状 | native µs | EAGER+ADP 54bit µs | 比 | FIXED 55bit µs | 比 |
|---|---|---:|---:|---:|---:|---:|
| x | `1024 × 1048576 × 1024` | 56969 | 114129 | **2.00** | 28183 | **0.495** |
| y | `1024³` batched ×1024（B=`D1D_tr`, stride 0） | 56521 | 152978 | **2.71** | 79645 | **1.41** |
| z | `1048576 × 1024 × 1024` | 56491 | 118179 | **2.09** | 33523 | **0.593** |
| 合計 | 3 方向 | 169981 | 385286 | **2.27** | 141351 | **0.831** |

native の device tendency 194.06 ms/stage のうち volume GEMM は約 170 ms、残り
約 24 ms は flux / 面 / assembly で emulation の対象外。ADP 予測は
385+24 = **409 ms/stage ≈ 2.11×**。§4 の遅い方の run（407.6 ms/stage、
1223 ms/step）と一致する。速い方の run（268.7 ms/stage）は内部 allocation の
揺らぎで、カーネルが 1.74× で安定しているわけではない。

### 6.3 x GEMM の mantissa 掃引（EAGER、FIXED）

IEEE FP64 は 53 bit。`mantissaBitCount = 8s−1` なので 55 bit は 7 スライス、
63 bit は 8 スライス。

| bits | x µs | TFLOP/s | vs native 56969 µs |
|---:|---:|---:|---:|
| 23 | 13389 | 164.3 | 0.235×（精度は FP64 相当ではない） |
| 39 | 19957 | 110.2 | 0.350× |
| 47 | 23851 | 92.2 | 0.419× |
| **55** | **28183** | **78.0** | **0.495×** |
| 63 | 58869 | 37.4 | 1.03×（損益分岐） |
| 79 | 76364 | 28.8 | 1.34× |
| ADP 54（DYNAMIC） | 114129 | 19.3 | 2.00× |

**交点は 63 bit。** ADP が選んだ 54 bit は 55 bit と同じ 7 スライスのはずなのに
4 倍遅い。したがって p=1023 の EAGER 負けは「bit が多すぎる」ではなく
**DYNAMIC 経路の解析オーバーヘッド**である。FIXED 55 なら x と z は勝ち、
3 本合計でも 0.831× まで下がる。リポジトリは精度比較のために ADP 既定を
変えていない。

### 6.4 正方 8192³（NVIDIA heatmap 側）

env=0 の native は 28816 µs（38.16 TFLOP/s）。env=1 の FIXED 55 は 9594 µs
（114.6 TFLOP/s）で **3.0 倍速い**。論文・ブログが示す勝ち領域はこちらで、
DG の K=Nq=1024・y の 1024³ batch ではない。K を Nq から切り離して深くする
ことは `AGENTS.md` の配列インタフェースではできない。

### 6.5 再現

```bash
module load nvhpc/26.3
cd bench_ozaki2
nvcc -O3 -arch=sm_100 -std=c++17 -o p1023_emu_why p1023_emu_why.cu -lcublas

# native
env -u CUBLAS_EMULATE_DOUBLE_PRECISION -u CUBLAS_EMULATION_STRATEGY \
  CUDA_VISIBLE_DEVICES=1 ./p1023_emu_why --no-int8 --no-square

# EAGER ADP / FIXED sweep
CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager \
  CUDA_VISIBLE_DEVICES=1 CUDA_MODULE_LOADING=EAGER ./p1023_emu_why
```

## 7. 本体を FIXED 55 に固定（2026-08-29）

`CublasEmulation=.true.` の既定は ADP ではなく、cuBLAS FIXED モードで
**55 mantissa bits** である。根拠は §6: 出力を FP64 相当にする幅は入力で変える
理由がなく、ADP の入力解析が p=1023 の x を FIXED 55 の 4 倍遅くしていた。

namelist で切り替えられる:

```fortran
CublasEmulation = .true.
EmulationMantissaControl = "FIXED"    ! or "DYNAMIC" / "ADP"
EmulationMantissaBits = 55
```

FIXED は毎回 55 bit。DYNAMIC は ADP で、`EmulationMantissaBits` は上限のみ。
同じ `EmulationMantissaControl` が OZAKI1 の残差早期打ち切りと OZAKI2 の A 残差
パック（最大 4）にも効く。スライス／法の本数は別ノブ
`OzakiSliceCount`（既定 7）と `OzakiModuliCount`（既定 7）。

実装: `cuda_cublas_gemm.cu` が有効時に
`CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH`、EAGER、
`CUDA_EMULATION_MANTISSA_CONTROL_FIXED` または `_DYNAMIC`、
`cublasSetFixedPointEmulationMaxMantissaBitCount(bits)` を設定する。
FIXED では起動前に効かせるため `CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT`
も `cublasCreate` の前に `setenv` する。起動ログは
`cuBLAS FP64 emulation: EAGER, FIXED 55 mantissa bits (no ADP)` または
`... DYNAMIC (ADP) max 55 mantissa bits`。

NVIDIA の文書どおり、FIXED は「すべての入力で native 以上」を保証しない。
このリポジトリの既定は ADP に任せず、FP64 クラスの既定幅 55 で比較する。

## 8. 本体 FIXED 55 vs native（2026-08-29、login GPU 1）

§4 の EAGER 列は ADP（DYNAMIC）だった。同じ入力族（`CublasEmulation` 以外同一、
`CUDA_MODULE_LOADING=EAGER`、実行ファイルは namelist 切替前の FIXED 55 バイナリ）
で 3 run 中央値を取り直した。device 時間は
`CUDA device GEMM tendency / (measured_steps × 3)`、すなわち 1 RK stage。

| p | Ne | measured steps | native ms/stage | FIXED 55 ms/stage | 比 | 旧 ADP 比（§4、/step） |
|---|---:|---:|---:|---:|---:|---|
| 7 | 32³ | 9 | 1.667 | 509.189 | **305×** | 131× |
| 255 | 1 | 18 | 1.078 | 2.166 | **2.01×** | 8.80× |
| 511 | 1 | 18 | 13.318 | 12.637 | **0.95×** | 1.50× |
| 575 | 1 | 45 | 20.753 | 27.177 | **1.31×** | 2.10× |
| 767 | 1 | 25 | 62.341 | 56.929 | **0.91×** | 1.94× |
| 1023 | 1 | 12 | 193.115 | OOM | — | 1.74× |

p=7 が ADP よりさらに遅いのは、K=8 に対して 55 bit（7 スライス）を常に全部
使うため。p=255 は ADP の 8.80× から 2.01× まで縮む。p=511 と p=767 では
FIXED 55 が native より速い。p=575 だけ 1.31× で負けが残る（ADP の 2.10× より
は改善）。

p=1023 native は 1 run（579.345 ms/step、旧 ADP 調査の 582.2 と一致）。
FIXED 55 は `!$acc data` の場コピー中に OOM。失敗時の
`total/free = 197568495616 / 7160791040`（約 184 GiB 使用、7.16 GiB 空き）に対し
次の確保が 8 GiB。永続 emulation workspace（8 GiB）を `init` で先に取るため、
点ごとの場（各配列が 1024³×8 B = 8 GiB）の最後が乗らない。孤立 GEMM ベンチは
この 176 GiB 級の場を持たない。ADP 当時の p=1023 emulation は workspace を
後から取れた run があり、2 run 中央値 1.74× として §4 に残っている。
