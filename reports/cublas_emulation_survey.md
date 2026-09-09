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

## 9. 現行ツリーで §8 を取り直した —— **表はそのまま成立する**。ただし精度は roundoff 相当ではない（2026-09-07、Slurm `84461`）

`measurement_inventory.md` §18.10 / §19.7 は、**公表軸のうち現行ツリーで
取り直していない唯一のもの**として `CublasEmulation` を挙げていた。取り直した。

**動機は「古いから」ではなく、§8 のあとに 2 つのコミットがこの表に効いた
ことである**: **`3d4045c`** は GEMM 系を全次数で動かした（すべての比の分母）、
**`941b066`** は `CublasEmulation` の **±Inf 数値バグ**を直したが、
**その修正のあとに速度を取り直していなかった。**

### 9.0 測定条件

| | |
|---|---|
| ジョブ | Slurm **`84461`**、node **`c191`**、**`--exclusive`**、16:23 |
| tree | **`663df57`**（ソースは `8547b07` 以降不変） |
| 実行ファイル | 凍結コピー `scale-dg_extraction.s33` |
| 入力 | `namelists/perf_p{7,255,511,575,767}_gemm.conf` から **`CublasEmulation` だけ**を差し替え。`Ne` / `PolyOrder` / `dt` / `nstep` は不変 |
| ラウンド | p=7 / 255 / 511 は 6、p=575 / 767 は 3、**native と emulation を交互** |
| 物差し | `CUDA device GEMM tendency` ÷ (計時 step 数 × 3) ＝ ms/stage。**§8 と同じ** |
| emulation 設定 | 既定の **FIXED 55 mantissa bits**（起動ログ `EAGER, FIXED 55 mantissa bits (no ADP)`） |
| 予測 | `output/s33/PREDICTIONS.txt` に投入前に凍結 |

**入力は §8 と同一である** —— `perf_p7_gemm.conf` は `NeX=32`（§8 の `Ne=32³`）、
他 4 次数は `Ne=1`。**したがって絶対値も比も直接比較できる。**

> **本節の数値検証は、ジョブ内で走らせた版が誤っていたため取り直した。**
> `SCALE_DG_DUMP_DQDT` は 1 行 1 値の**テキスト**だが、ジョブ内の比較スクリプトが
> それをバイナリの倍精度列として読んでいた（3 次数で同一の異常値
> `max|diff|=1.092e+25` を返したことで判明）。**下表はログインノードで
> `jobs/tsubame_paths.sh` の `cmp_dump` と同じ読み方で取り直したものである。**
> 計時側はこの誤りの影響を受けない。

### 9.1 §8 の表は現行ツリーでそのまま成立する

| p | native ms/stage | §8 native | 差 | FIXED 55 ms/stage | §8 FIXED 55 | 差 | **比** | **§8 の比** | **比の変化** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 1.6172 | 1.667 | −2.99% | 481.6947 | 509.189 | −5.40% | **297.9** | 305 | **−2.3%** |
| 255 | 1.0533 | 1.078 | −2.29% | 2.1640 | 2.166 | −0.09% | **2.055** | 2.01 | +2.2% |
| 511 | 13.2762 | 13.318 | −0.31% | 12.5546 | 12.637 | −0.65% | **0.946** | 0.95 | −0.5% |
| 575 | 20.6941 | 20.753 | −0.28% | 27.5759 | 27.177 | +1.47% | **1.333** | 1.31 | +1.7% |
| 767 | 63.2985 | 62.341 | +1.54% | 57.9500 | 56.929 | +1.79% | **0.916** | 0.91 | +0.6% |

レンジ（ms/stage、min–max）は全次数で非重複:

| p | native | FIXED 55 |
|---:|---|---|
| 7 | 1.6168–1.6177 | 478.5333–485.9930 |
| 255 | 1.0528–1.0542 | 2.1603–2.1655 |
| 511 | 13.2707–13.2781 | 12.5474–12.5819 |
| 575 | 20.6926–20.6950 | 27.4756–27.8581 |
| 767 | 63.2976–63.2985 | 57.8758–58.1585 |

**登録した予測 V4（少なくとも 1 次数で比が 10% 超動く）は外れた。**
**5 次数とも −2.3%〜+2.2% で再現する。** 絶対値も −5.4%〜+1.8% に収まる。
**したがってこの軸は `3d4045c` と `941b066` をまたいで安定であり、
`measurement_inventory.md` §19.7 は §8 をそのまま引いてよい。**

V2（p=7 の比 > 100）と V3（p=511 / p=767 が 1.05 以下）も成立した。

### 9.2 数値検証 —— **Inf は出ない。しかし roundoff 相当でもない**

`SCALE_DG_VARYING_COEFF=1`、owned `dqdt` 全点、対照は p=7 が
`CUDAFORTRAN_SPLIT`、p≥255 が `CUDAFORTRAN_GEMM`（どちらも native FP64）:

| p | 点数 | max abs | **相対** | **Inf / NaN** |
|---:|---:|---:|---:|---:|
| 7 | 16 777 216 | 5.779e-13 | **6.761e-14** | **0** |
| 255 | 16 777 216 | 9.407e-11 | **1.100e-11** | **0** |
| 511 | 134 217 728 | 4.824e-10 | **5.642e-11** | **0** |

- **V1 は成立した。`941b066` の修正は効いている** —— 3 次数・1.68 億点で
  Inf も NaN も 1 つも出ない。
- **しかし FIXED 55 は roundoff 相当ではない。** 相対差は
  **6.8e-14 → 1.1e-11 → 5.6e-11** と**還元深さ `K = Nq` とともに増える**。
  `measurement_inventory.md` §18.4 が記録した**全 CUDA 経路の最悪値
  2.113e-13** に対し、p=255 で **50 倍**、p=511 で **270 倍**である。
- **これは実装の欠陥ではなく設計どおりである**（55 bit の固定小数点で
  FP64 を近似する）。**記録すべきなのは、`CublasEmulation` を
  「数値的に等価な高速化」として扱ってはいけない**という点であり、
  速度を引くときは必ずこの表を一緒に読むこと。
- §5 の p=255 **7.461e-14** とは**直接比較できない** —— あれは ADP
  （DYNAMIC）時代の値で、§7 が本体を FIXED 55 に固定する前のものである。
  §5 の p=7 の「ビット一致」も `Ne=2³`（4 096 点）での測定で、本節の
  `Ne=32³`（1 677 万点）とは別の問題である。**§5 の表は書き換えない。**

### 9.3 本節でも測っていないこと（推測で埋めない）

- **p=1023。** §8 と同じく emulation の 8 GiB 永続 workspace が場のコピーに
  乗らない。**本節は試していない**（§8 の OOM が現行ツリーでも起きるかは
  未確認）。
- **`EmulationMantissaControl = "DYNAMIC"`（ADP）の現行値。**
  §4 / §5 のまま。本節は既定の FIXED 55 だけである。
- **H100 / A100 での `CublasEmulation`。** `measurement_inventory.md` §19.7 の
  とおり、3 台のうち GB200 でしか測っていない。本節も GB200 だけである。
- **精度が `K` とともに悪化する係数。** 3 点（6.8e-14 / 1.1e-11 / 5.6e-11）
  しかなく、関数形を当てはめていない。

## 10. 3 台で測った —— **emulation が勝つのは GB200 だけである**。ADP と p=1023 も現行ツリーで取った（2026-09-08、Slurm `86678` / SGE `8599546` / TCS `9545645`）

### 10.0 測定条件

| | GB200 | H100 | A100 |
|---|---|---|---|
| ジョブ | Slurm **`86678`** | SGE **`8599546`** | TCS **`9545645`** |
| ノード | `c384`（`--exclusive`） | `r17n10`（`gpu_1=1`） | `wa34`（`gpu=1`） |
| GPU | NVIDIA GB200 189 471 MiB | NVIDIA H100 95 830 MiB | A100-SXM4-**40GB** |
| commit | `094aabe` | `029e82b` | `0e64b15` |
| コンパイラ | nvfortran 26.3 | nvfortran 26.1 | nvfortran 25.9 |
| cuBLAS（ヘッダ） | **13.2.1** | **13.2.0** | **12.9.1** |
| ラウンド | 6（p=7/255/511）・3（p=575/767/1023） | 同左（1023 無し） | 3 |
| 物差し | `CUDA device GEMM tendency` ÷ (計時 step 数 × 3) ＝ ms/stage。**§8 / §9 と同じ** |

- **3 commit はソース同一**である（`*.f90` / `*.cuf` / `*.cu` / `*.h` /
  `*.erb` / `Makefile` / `namelists/` の差分が空）。差はレポートと
  `jobs/` だけで、3 列は 1 本のソースの測定である。
- **3 列は 1 本のスクリプト**（`jobs/cublas_emulation_ab.sh`）で、機械差は
  ラッパ（`{rikyu,tsubame,wisteria}_cublas_emu.sh`）だけが持つ。入力は
  `namelists/perf_p{order}_gemm.conf` から emulation ノブだけを差し替えたもので、
  `Ne` / `PolyOrder` / `dt` / `nstep` は不変。
- **数値検証はテキスト読みの `cmp_dump`**（`jobs/tsubame_paths.sh` と同じ）。
  §9.0 の箱書きが記録した「ジョブ内でバイナリとして読んでいた」欠陥は、
  本体を共有したことで構造的に解消した。
- 予測は投入前に `output/s39/PREDICTIONS.txt`（GB200、V1–V8）と
  `output/s41/PREDICTIONS.txt`（H100 / A100、E1–E7）に凍結した。

### 10.1 GB200 —— §9 の表は再現する（V1 / V2 成立）

| p | native ms/stage | §9.1 native | 差 | FIXED 55 | §9.1 FIXED 55 | 差 | **比** | **§9.1 の比** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 1.6130 | 1.6172 | −0.26% | 484.02 | 481.69 | +0.48% | **300.1** | 297.9 |
| 255 | 1.0505 | 1.0533 | −0.27% | 2.1636 | 2.1640 | −0.02% | **2.060** | 2.055 |
| 511 | 13.2649 | 13.2762 | −0.09% | 12.5570 | 12.5546 | +0.02% | **0.9466** | 0.946 |
| 575 | 20.6899 | 20.6941 | −0.02% | 27.8431 | 27.5759 | +0.97% | **1.346** | 1.333 |
| 767 | 63.2833 | 63.2985 | −0.02% | 57.8758 | 57.9500 | −0.13% | **0.9145** | 0.916 |

**`663df57` → `094aabe` でこの軸は動いていない。** したがって以下の H100 列は
§9 の GB200 列とそのまま比較してよい。

### 10.2 H100 —— **どの次数でも emulation が負ける**（予測 E3 / E4 は外れ）

| p | native ms/stage | FIXED 55 ms/stage | **emu/nat** | GB200 の同じ比（§10.1） |
|---:|---:|---:|---:|---:|
| 7 | 2.7773 | 694.24 | **250.0** | 300.1 |
| 255 | 1.4756 | 4.3991 | **2.981** | 2.060 |
| 511 | 14.6866 | 32.013 | **2.180** | **0.9466** |
| 575 | 21.963 | 69.687 | **3.173** | 1.346 |
| 767 | 62.419 | 154.21 | **2.471** | **0.9145** |

30 ラン全 `rc=0`、レンジは全次数で非重複（例: p=511 native 14.6818–14.6949 対
FIXED 55 31.8163–32.1930）。

**`GB200 では p=511 / 767 で emulation が native を上回る` という §8 / §9 の
結論は、機械の性質であって FP64 emulation 一般の性質ではない。**
論文が emulation を「大次数で勝つ」と書くなら、**GB200 に限る**と書かなければ
ならない。

**差は emulated 側にある。** p=511 の native は H100 14.6866 対 GB200 13.2649
で **+10.7%** しか違わないのに、FIXED 55 は 32.013 対 12.5570 で **2.55 倍**
違う。native の機械差では説明できない。

**機構は未測定である。** 候補は「INT8 テンソル演算のピークが FP64 ピークに
対して GB200 のほうが厚い」だが、**本節はどちらの機械でも emulation の
カーネル内訳を取っていない**ので、公称ピーク比を引いて説明した気になっては
いけない。閉じるには `jobs/cublas_kernel_names_nsys.sh` を両機で走らせて、
emulated GEMM の中身（INT8 mma / 分割 / 復元）に時間を割り付ける必要がある。

### 10.3 A100 —— **測れない。ハードではなくツールチェインが理由である**（予測 E1 のリスク側）

`cuda_cublas_gemm.cu` は emulation 呼び出しを `CUBLAS_VERSION >= 130002`
（cuBLAS 13.0 update 2）で囲っている。Aquarius の最新モジュール
`nvidia/25.9` が持つ cuBLAS は **12.9.1** で、この線に届かない。したがって
`#else` 側がコンパイルされ、emulated run は全部こうなる:

```
cuBLAS FP64 emulation requires cuBLAS 13.0 update 2 or newer
ERROR STOP  cublas_gemm_init: unrecognized error code
```

9 ラン（p=7/255/511 × 3 ラウンド）と検証 2 本、**すべて同じ形で停止**した。

- **これは「A100 では emulation が動かない」ではない。** sm_80 が
  cuBLAS 13.0u2 の emulation に対応するかどうかは、**本測定は何も言っていない。**
  `OZAKI2` の sm_90 / sm_80 欠陥（`ozaki2_implementation_report.md` §8.7–§8.13）は
  ソースの欠陥だったが、こちらは環境の版であり、**別種である**。
- **同じジョブの native 側は 9/9 成功**しており、A100 列の交差検証になる:
  p=7 5.4935 / p=255 3.0431 / p=511 34.6535 ms/stage。
- 他の A100 列はすべて `nvidia/25.9` で測っている（`jobs/wisteria_paths.sh`）。
  新しい cuBLAS を持ってきて埋めた値は同じ表の行としては読めないので、
  やるなら「軸外の存在確認」として別枠になる —— **が、その逃げ道も無い。**
- **Aquarius には cuBLAS 13 系が存在しない。** ログインノードの `module av`
  （2026-09-08）で提供されるのは `nvidia/{21.3, 22.2, 22.5, 22.7, 22.9,
  22.11, 23.3, 23.9, 23.11, 24.1, 24.11, 25.9}` と `cuda/{10.1 … 12.2, 12.6}`
  であり、**最も新しい `nvidia/25.9` の 12.9.1 が上限、`cuda` 系は 12.6 が
  上限**である。したがって **A100 の `CublasEmulation` セルは、この機械の
  ソフトウェア環境が更新されるまで測定不能**である。推測で埋めない。

### 10.4 ADP（`EmulationMantissaControl = "DYNAMIC"`）—— **小次数で勝ち、大次数で負ける。しかもビット一致**（§9.3 の 2 件目）

GB200、同じジョブ。**ADP は §4 / §5 以来はじめて現行ツリーで測られた。**

| p | native | FIXED 55 | ADP | **adp/nat** | **adp/emu** |
|---:|---:|---:|---:|---:|---:|
| 7 | 1.6130 | 484.02 | 168.28 | 104.3 | **0.348** |
| 255 | 1.0505 | 2.1636 | 1.8537 | 1.765 | **0.857** |
| 511 | 13.2649 | 12.5570 | 18.9625 | 1.430 | **1.510** |
| 575 | 20.6899 | 27.8431 | 31.6532 | 1.530 | **1.137** |
| 767 | 63.2833 | 57.8758 | 83.1912 | 1.315 | **1.437** |
| 1023 | 196.66 | 135.94 | 244.80 | 1.245 | **1.801** |

- **p=7 で ADP は FIXED 55 の 2.88 倍速い**（予測 V3 成立）。
  **p=255 でも 14% 速い。しかし p=511 以上では 1.14〜1.80 倍遅い**
  （予測 V4「±10% 以内」は **外れ**）。**交代は p=255 と p=511 の間にある。**
- **精度では ADP が FIXED 55 を支配する。** 点変化係数・全所有点の比較で
  **p=7 / 255 / 511 とも参照実装とビット一致（相対差 0.000e+00）**、
  一方 FIXED 55 は 6.761e-14 / 1.100e-11 / 5.642e-11（§9.2 と 4 桁一致）。
  **予測 V7 成立。**
- **したがって既定の FIXED 55 は、p≤255 では遅くて精度も悪い。**
  既定値そのものは本節では変えない（`CublasEmulation` は本体の生産経路では
  なく、`AGENTS.md` の経路定義にも入らない調査軸である）が、
  **§7 が FIXED 55 を既定にした判断は、次数を通した比較では成り立たない。**
- **ADP はメモリを余計に使う。** p=1023 の ADP は 3 ラウンド中 **2 本が
  `cuMemAlloc` の OOM**（1 本だけ成功して 244.80 ms/stage）。同じ次数の
  FIXED 55 は 3/3 成功しており、**ADP 固有の追加確保である。**

### 10.5 p=1023 の emulation は **動く**（§9.3 の 1 件目、予測 V5 は外れ）

§8 / §9 は「8 GiB 永続 workspace が場のコピーに乗らない」として試していな
かった。**現行ツリーでは動き、しかも native より速い。**

| p=1023 | ms/stage（3 ラウンド中央値） | レンジ | 対 native |
|---|---:|---|---:|
| native | 196.658 | 196.623–196.660 | 1 |
| FIXED 55 | **135.936** | **130.034–189.975** | **0.691** |
| ADP | 244.799 | 1 本のみ（他 2 本 OOM） | 1.245 |

- **emulation の勝ち幅は次数とともに広がる**: 0.947（511）→ 0.914（767）
  → **0.691（1023）**。
- **ただし p=1023 の FIXED 55 は荒れている**（130.0 / 約 136 / 190.0）。
  native は 3 本が 0.02% に収まるので、**荒れは emulated 側の性質である。**
  原因は未測定。ワークスペース確保と場の 176 GiB がカードの端で競合している
  こと（ADP が同じ次数で OOM すること）を思えば、**メモリ端の効果を疑うのが
  自然だが、確かめていない。**

### 10.6 予測の当たり外れ

| 予測 | 結果 |
|---|---|
| V1 §9.1 の比が ±5% で再現 | **成立**（−0.5%〜+1.0%） |
| V2 native が ±2% で再現 | **成立**（−0.27%〜−0.02%） |
| V3 p=7 で ADP > 2× FIXED | **成立**（2.88×） |
| V4 p=255 / 511 で ADP と FIXED が ±10% | **外れ**（p=255 −14%、**p=511 +51%**） |
| V5 p=1023 emulated は OOM | **外れ**（動作し、native の 0.691×） |
| V6 FIXED 55 の相対誤差が §9.2 を再現 | **成立**（4 桁一致） |
| V7 ADP の誤差 ≤ FIXED | **成立**（3 次数ともビット一致） |
| V8 Inf / NaN なし | **成立** |
| E1 H100 は動き、A100 はツールチェイン次第 | **H100 成立、A100 は 12.9.1 で不可** |
| E2 p=7 の比が両機で 100 超 | **H100 成立**（250） |
| E3 p=511 で両機とも比 < 1 | **外れ**（H100 は 2.18） |
| E4 A100 の比 < H100 の比 | **測れず**（A100 が走らない） |
| E5 相対誤差が GB200 と 2 倍以内 | **成立**（4 桁一致） |
| E7 Inf / NaN なし | **成立** |

### 10.7 本節でも測っていないこと

- **H100 と GB200 で emulated 側が 2.55 倍違う機構。** カーネル内訳を
  どちらの機械でも取っていない（§10.2）。
- **A100 の emulation セルそのもの。** cuBLAS 13.0u2 以上が Aquarius に
  入るまで測れない（§10.3）。
- **p=1023 FIXED 55 の 46% のばらつき**（§10.5）。3 本しかない。
- **ADP が p=511 以上で遅くなる理由。** 選ばれた仮数ビット数を実行時に
  取り出していない。§6 は p=1023 で 54 bit を選ぶと記録しているが、
  それは別ツリーの測定である。
- **ADP の追加メモリ量。** p=1023 で OOM することは分かったが、
  何バイト余計に要るかは測っていない。
- **H100 / A100 の ADP。** 本節の ADP は GB200 だけである。

## 11. emulated GEMM の中身（GB200 半分、2026-09-08、Slurm `86730`、node `c384`）—— **主カーネルは `sm100` 専用の fused IMMA である**

§10.2 が「H100 と GB200 で emulated 側だけが 2.55 倍違う。機構は未測定」と
書いた件の GB200 側。`jobs/cublas_kernel_names_nsys.sh` に `KINDS` ノブを足し、
`GEMM`（native）と `GEMM_EMU`（`CublasEmulation = .true.`、同じ
`CUDAFORTRAN_GEMM`）を **同一ジョブで交互に** トレースした。
`nstep=10`、`UseCudaGraph = .false.`（`nsys` はグラフ経路を測れない）、
`--resolve-symbols=false` ＋ `DEBUGINFOD_URLS=`。4 トレースとも
`profile rc=0` / `stats rc=0`。予測は `output/s43/PREDICTIONS.txt`。

### 11.1 emulation は 1 カーネルではなく 3 役である（予測 N1 成立）

p=511、GPU 時間の内訳（30 ステージ ＝ 体積 GEMM 90 本）:

| カーネル | 本数 | 合計 [ms] | 1 本 [µs] | 割合 |
|---|---:|---:|---:|---:|
| `cublasLt_fused_imma_dgemm_kernel_sm100_cga2x1` | 90 | **201.128** | 2 234.8 | 51.6% |
| `cublasLt_imma_emu::max_scale_pack_ker<64,4,128,…>` | **180** | **100.431** | 557.9 | 25.8% |
| `cublasLt_imma_emu::cutlass_80_tensorop_d884gemm_128x128_16x3` | 90 | — | — | **0.3%** |
| `separable_lift_assembly`（本実装） | 30 | 34.499 | 1 150.0 | 8.9% |
| `volume_flux`（本実装） | 30 | 31.098 | 1 036.6 | 8.0% |

同じジョブの native:

| カーネル | 本数 | 合計 [ms] | 1 本 [µs] | 割合 |
|---|---:|---:|---:|---:|
| `cutlass_80_tensorop_d884gemm_64x128_16x3`（cuBLAS） | 90 | **327.340** | 3 637.1 | 78.9% |
| `separable_lift_assembly` | 30 | 34.370 | 1 145.7 | 8.3% |
| `volume_flux` | 30 | 31.100 | 1 036.7 | 7.5% |

- **pack は GEMM 1 本につき 2 本**（A 側と B 側）で、n=180 は 90 GEMM × 2 である。
- **周囲のカーネルは動いていない**（lift 34.370 → 34.499、flux 31.100 →
  31.098）。**差は体積 GEMM だけで起きている**ので、§10 の ms/stage 比を
  そのままこの内訳で説明できる。
- **予測 N4 も成立**: native の GB200 は `cutlass_80_tensorop_d884gemm`
  （`h100_report.md` §8.7.2 の縮退）で、**emulated の主カーネルはそれではない。**

### 11.2 勝敗の算術 —— **IMMA GEMM が 1.63 倍速く、pack がその勝ちを 3 割返す**

体積 GEMM の合計（emulated ＝ imma ＋ pack）:

| p | native [ms] | imma [ms] | pack [ms] | emulated 計 [ms] | **emu ÷ nat** | §10 の ms/stage 比 |
|---:|---:|---:|---:|---:|---:|---:|
| 255 | 23.748 | 38.457 | 15.254 | 53.711 | **2.262** | 2.060 |
| 511 | 327.340 | 201.128 | 100.431 | 301.559 | **0.921** | 0.9466 |

- **p=511 では IMMA GEMM 単体が native の 0.614 倍**（201.1 対 327.3）である。
  **pack が native の 0.307 倍を足し戻すので、正味 0.921 倍**に縮む。
- **p=255 では IMMA GEMM 単体がすでに native の 1.62 倍**（38.5 対 23.7）で、
  pack がさらに 0.64 倍を足す。**小さい `K` では INT8 に載せ替える価値が無い。**
- **pack のコストは GEMM 本体より緩やかにしか増えない** —— p=255 → 511 で
  native GEMM が 13.8 倍、imma が 5.2 倍、pack が 6.6 倍。§10.5 で
  emulation の勝ち幅が次数とともに広がる（0.947 → 0.914 → 0.691）のは
  この形と整合する。

### 11.3 まだ測っていない —— **H100 側**

**本節は GB200 だけである。** `jobs/tsubame_cublas_emu_nsys.sh` が同じ本体で
H100 を取るが、まだ走っていない。**したがって §10.2 の 2.55 倍は説明されて
いない。**

ただし本節は候補を 1 つ非常に強く示唆する: **GB200 の主カーネル名は
`..._sm100_cga2x1` であり、アーキテクチャ専用（Blackwell の thread block
cluster を使う）である。** sm_90 に同じ名前のカーネルは存在し得ないので、
**H100 は別のカーネルを踏んでいる可能性が高い。** そうであれば §10.2 の差は
「INT8 のスループット比」ではなく **「cuBLAS が sm_100 にだけ専用実装を
持っている」**という話になる。**これは推測であり、H100 のトレースが出る
まで結論にしない**（予測 N2 / N3 は H100 が無いと判定できない）。

## 12. H100 の ADP —— **FIXED 55 に全次数で勝つ。GB200 とは逆である**（2026-09-08、SGE `8602138`、node `r12n3`）

§10.4 は ADP を GB200 でしか測っていなかった。H100 を同じ本体・同じ
namelist・同じ 6/3 ラウンドで取った（45 ラン全 `rc=0`）。

| p | native | FIXED 55 | ADP | **adp/emu**（H100） | **adp/emu**（GB200 §10.4） |
|---:|---:|---:|---:|---:|---:|
| 7 | 2.7805 | 696.80 | 152.80 | **0.219** | 0.348 |
| 255 | 1.4732 | 4.3963 | 2.8755 | **0.654** | 0.857 |
| 511 | 14.5853 | 31.7626 | 27.7424 | **0.873** | **1.510** |
| 575 | 21.8076 | 69.4893 | 44.4464 | **0.640** | 1.137 |
| 767 | 61.9685 | 153.489 | 117.169 | **0.763** | 1.437 |

- **H100 では ADP がどの次数でも FIXED 55 に勝つ。** GB200 では p=511 以上で
  1.14〜1.80 倍**負ける**（§10.4）。**仮数制御の優劣は機械で反転する。**
- **それでも emulation は native に勝てない**: `adp/nat` は p≥255 で
  **1.89〜2.04**、p=7 で 55.0。**ADP は H100 の赤字を半分にするだけである。**
  §10.2 の結論（H100 では emulation が負ける）は仮数モードを変えても動かない。
- **精度は ADP が支配する**（§10.4 と同じ）: p=7 / 255 は参照実装と
  **ビット一致**、p=511 は **2.078e-16**（GB200 は 0.000e+00）。
  FIXED 55 の 5.642e-11 に対して 5 桁良い。
- **同じセルの再現性**: `emu/nat` は §10.2（node `r17n10`、job `8599546`）と
  本節（node `r12n3`）で 2.981→2.984 / 2.180→2.178 / 3.173→3.186 /
  2.471→2.477 と **3 桁一致**する。native 側は −0.69%〜+0.12% 動く。
  **比のほうがノードをまたいで安定である。**

### 12.1 H100 の emulated カーネル内訳は **まだ無い**（手順上の失敗）

§11.3 が求めた H100 のトレース（SGE `8602137`）は走ったが、
**`KINDS` ノブを持たない古い `jobs/cublas_kernel_names_nsys.sh` に対して
投入されたため、`GEMM` と `GEMM_CUTE`（既定）をトレースして終わった。**
emulated 経路は 1 本も採れていない。**結果自体は正常終了しており、
中身を見るまで正しい測定に見える**という点でたちが悪い。

対策として `jobs/{rikyu,tsubame}_cublas_emu_nsys.sh` は投入時に本体へ
`readonly KINDS=` があるかを確認し、無ければ**その場で異常終了する**ように
した。**したがって §10.2 の 2.55 倍は依然として未説明であり、§11.3 の
「H100 は別のカーネルを踏んでいるのではないか」という示唆も未検証のままである。**

なお副産物として、この誤投入は H100 の **native / `GEMM_CUTE`** の内訳を
現行ツリーで与えている（p=255 native は `sm90_xmma_gemm_f64f64_..._tensor16x8x8`
の 3 タイル、`GEMM_CUTE` は `MmaMultistage` 系 3 本）。
**`h100_report.md` §8.7.2 の「H100 の cuBLAS は手書き sm_90 xmma」は現行
ツリーでも成立している。**

## 13. H100 の emulated カーネル内訳 —— **§10.2 の差は説明された。IMMA GEMM が H100 で 2.87 倍遅く、native は逆に H100 が速い**（2026-09-08、SGE `8602308`、node は job ログ参照）

§11.3 / §12.1 が残した H100 側。**§12.1 の誤投入を直したうえで取り直した**
（本体に `KINDS` があることをラッパが確認する）。4 トレースとも
`profile rc=0` / `stats rc=0`。

### 13.1 役は同じ、カーネルはアーキテクチャ専用（予測 N1 / N2 成立）

H100 の emulated 経路も **3 役**である:

| 役 | GB200 | H100 |
|---|---|---|
| 融合 IMMA GEMM | `cublasLt_fused_imma_dgemm_kernel_`**`sm100`**`_cga2x1` | `cublasLt_fused_imma_dgemm_kernel_`**`sm90`** |
| pack（GEMM 1 本に 2 本） | `cublasLt_imma_emu::max_scale_pack_ker<64,4,128,…>` | 同名 |
| 誤差項 | `cublasLt_imma_emu::cutlass_80_tensorop_d884gemm_128x128_16x3` | 同名 |

**アルゴリズムは同じで、主カーネルだけがアーキテクチャ別に用意されている。**
§11.3 の「H100 は別のカーネルを踏んでいるのではないか」は**そのとおりだった
が、含意は逆**である —— 専用カーネルが**無い**のではなく、**sm_90 用も
ちゃんとある**。差は実装の有無ではなく速度である。

### 13.2 体積 GEMM の内訳（p=511、30 ステージ ＝ 90 本）

| 量 [ms] | GB200 | H100 | **H100 ÷ GB200** |
|---|---:|---:|---:|
| native 体積 GEMM 合計 | 327.340 | **221.696** | **0.677**（H100 が速い） |
| 融合 IMMA GEMM | 201.128 | **576.378** | **2.866** |
| pack ×180 | 100.431 | 153.252 | 1.526 |
| 誤差項 | 1.252 | 1.557 | 1.244 |
| **emulated 合計** | **302.811** | **731.187** | **2.415** |
| **emulated ÷ native** | **0.925** | **3.298** | — |

p=255 も同じ向きである（native 23.749 → 16.960 で H100 が **0.714 倍**、
IMMA GEMM 38.457 → 78.050 で **2.030 倍**、pack 1.537 倍）。**予測 N5 成立:
2 つの次数が同じカーネルを犯人に指す。**

### 13.3 **これで §10.2 は閉じた。差は 2 つの逆向きの事実の積である**

1. **emulated 側**: 融合 IMMA GEMM が **H100 で 2.87 倍遅い**。
2. **native 側**: FP64 GEMM は **H100 のほうが 1.48 倍速い**（0.677 の逆数）。
   **GB200 の FP64 Tensor ピークは CUDA core と同じ 40.1 TFLOP/s だが、
   H100 は 66.9 TFLOP/s** である（`AGENTS.md` の屋根表）。**native の分母が
   H100 では高い。**

**両方が同じ向きに効く**: 体積 GEMM の `emu/nat` は 0.925（GB200）対
3.298（H100）で **3.57 倍**違い、うち native 側が 1.48 倍、emulated 側が
2.42 倍を担う。**「GB200 では emulation が勝つ」は、Blackwell の INT8 経路が
速いことと、Blackwell の FP64 Tensor の屋根が低いことの、両方の帰結である。**

**予測 N3 は半分外れた**: 「IMMA GEMM の**割合**が両機で 10 ポイント以内」は
66.4% 対 78.8% で **12.4 ポイント**あり成立しない。しかし「赤字は IMMA GEMM
に集中し、pack ではない」という主要部分は成立している（2.87 対 1.53）。

### 13.4 ステージ時間との突き合わせ（内訳が §10 の比を再現する）

周囲のカーネルは native と emulated で動かないので、
`(GEMM 合計 ＋ 周囲) ÷ (native GEMM 合計 ＋ 周囲)` が ms/stage 比になるはずである:

| 機械 | 周囲 [ms] | 予測される比 | §10 の実測比（p=511） |
|---|---:|---:|---:|
| GB200 | 65.5 | (302.8+65.6)/(327.3+65.5) = **0.938** | **0.9466** |
| H100 | 207.6 | (731.2+207.6)/(221.7+207.6) = **2.187** | **2.178** |

> **脚注（2026-09-09 追記、表は未変更）: GB200 行の「周囲」列の 65.5 と、
> 同じ行の式の分子にある 65.6 は別の測定であり、取り違えではない。**
> 周囲のカーネルは native 実行と emulated 実行のそれぞれで別に計時されている
> （§11.1 の 2 表）。**65.5 は native 実行側**（`separable_lift_assembly`
> 34.370 ＋ `volume_flux` 31.100 ＝ 65.470）、**65.6 は emulated 実行側**
> （34.499 ＋ 31.098 ＝ 65.597）である。式
> `(302.8+65.6)/(327.3+65.5)` は分子に emulated 側、分母に native 側を
> 使っており、列見出しの「周囲」は**そのうち native 側だけ**を載せている。
> 差 0.13 ms（0.2%）は §11.1 の「周囲のカーネルは動いていない」と整合する。
> H100 行は分子・分母とも 207.6 を使っているが、これも 1 つの測定を両側に
> 流用したのではなく、**両実行の値が 0.04 ms（0.02%）しか違わない**ためで
> ある（native 105.431 ＋ 102.147 ＝ 207.578 ms、emulated
> 105.384 ＋ 102.153 ＝ 207.537 ms、TSUBAME ジョブ `8602308` の
> `p511_GEMM{,_EMU}_cuda_gpu_kern_sum.csv`）。どちらの行も、両側を native
> 側の値に揃えても表の 3 桁は変わらない（GB200 0.9379 → 0.9376、
> H100 2.18677 → 2.18687）。

**2 機械とも 1% 以内で一致する。** §10 の ms/stage 表と §13 のカーネル内訳は
同じものを別の物差しで見ている。

**H100 の周囲のカーネルが GB200 の 3 倍遅い**ことも同時に見えている
（`separable_lift_assembly` 105.4 対 34.5 ms、`volume_flux` 102.1 対 31.1 ms）。
**これらは DRAM 律速で、機械差は帯域の差である。** 本節の主題ではないが、
**H100 で native の体積 GEMM が速いのに tendency 全体では GB200 が勝つ**のは
これが理由であり、`measurement_inventory.md` §19 の「device 軸と stage 軸で
機械間比が違う」という観測と同じ現象である。

## 14. 精度を「勝つ次数」まで伸ばした —— p=767 を足し、p=1023 は**測れないことが結果である**（2026-09-09、Slurm `88634`、node `c181`）

### 14.1 なぜ

`draft_appendix_a4.md` A.4.3 が自分で名乗っている穴:
*of the three orders at which we count the emulated path as faster, accuracy
was measured at only the lowest*。emulation が native に勝つのは
**p = 511 / 767 / 1023** の 3 次数（§10.1、§10.5）だが、相対差を測ったのは
**p = 7 / 255 / 511**（§9.2）で、勝つ次数のうち最も低い 1 つだけだった。

### 14.2 条件

commit `e19c834`、Slurm `88634`、node `c181`、`--exclusive`、
`jobs/acc_highorder_emu_ozaki.sh`。`SCALE_DG_VARYING_COEFF=1`、`nstep=1`、
owned `dqdt` の**全点**をテキストにダンプし、**同一次数の native cuBLAS
`CUDAFORTRAN_GEMM`** と突き合わせた。namelist は
`namelists/perf_p{767,1023}_gemm.conf` から emulation ノブだけを差し替えたもの。

### 14.3 結果 —— K とともに悪化する系列は p=767 まで続く

| p | Nq | 比較点数 | max abs | ref max | **相対差** | Inf/NaN |
|---:|---:|---:|---:|---:|---:|---:|
| 767 | 768 | 452 984 832 | 1.266e-09 | 8.549 | **1.481e-10** | 0 |

§9.2 の 3 点に本節を足すと:

| Nq | 8 | 256 | 512 | **768** |
|---|---:|---:|---:|---:|
| FIXED 55 の相対差 | 6.8e-14 | 1.1e-11 | 5.6e-11 | **1.481e-10** |

- **単調増加は続く。**512 → 768 は `K` が 1.5 倍で相対差が **2.63 倍**なので、
  **`K` に比例するより速い**。§9.3 の「関数形を当てはめていない」は
  4 点になっても維持する（当てはめない）。
- `measurement_inventory.md` §18.4 の**全 CUDA 経路の最悪値 2.113e-13**
  に対し、p=767 では **701 倍**である（p=511 の 270 倍から拡大）。
- **したがって A.4.3 の限定は「最も低い 1 つ」から「3 つのうち 2 つ」に
  縮む。**残る 1 つは下記のとおり測れない。

### 14.4 ADP は p=767 でも**ビット一致**である

| p | ADP（DYNAMIC、55 bit 上限） | max abs |
|---:|---|---:|
| 767 | native `GEMM` と **ビット一致** | **0.000e+00** |

§10.4 は p=7 / 255 / 511 の 3 次数でビット一致を報告していた。**4 次数目である。**
**速さと精度が設定で分離する**という §10.4 の結論 —— FIXED 55 は速いほうで
精度が足りず、ADP は精度が出るが速くない —— は p=767 でも成り立つ。

### 14.5 p=1023 の emulated は**点変化係数では測れない**（これは結果である）

| p | FIXED 55 | ADP |
|---:|---|---|
| 1023 | **OOM**（`cuMemAlloc` error 2、`main.f90:123`） | **OOM**（同上） |

失敗時のログ: `allocated block device: ... size:25769803776`（24 GiB）の直後、
**8 589 934 592 B（8 GiB）の確保で落ちる**。これは §8 が記録し §10.5 が
「現行ツリーでは動く」と上書きした、あの 8 GiB 永続 workspace である。

**§10.5 と矛盾しない。**§10.5 が p=1023 emulated を回せたのは
**定数係数のベンチマーク**（`namelists/perf_p1023_gemm.conf` そのまま)で、
本節は `SCALE_DG_VARYING_COEFF=1` —— `u,v,w,Escale,normal_fn,Fscale` を
**すべて点ごとの場として確保する** —— を足している。同一次数の native 参照は
本節でも通っているので、**足りないのは workspace 8 GiB ちょうどの分**である。
§10.5 自身が「FIXED 55 は 130.0 / 約 136 / 190.0 と荒れており、ワークスペースと
場の 176 GiB がカードの端で競合している」と書いた、その端である。

**したがって:**

- **p=1023 の emulated 経路は、この機械では数値検証できない。**
  「取っていない」ではなく「**取れない、その理由は測ってある**」。
- A.4.3 は *accuracy was measured at only the lowest of the three* を
  **「3 つのうち 2 つで測った。最高次は 186 GB の機械では点変化係数の場と
  8 GiB workspace が同居できないので測れない」**に書き換えられる。
- **§10.5 の p=1023 の速度値（0.691）は据え置く。**あれは定数係数の
  ベンチマークで取った値で、本節はその精度を否定していない —— 測っていない。

### 14.6 本節でも測っていないこと

- **p=1023 を定数係数で比べること。**やれば何か出るが、`AGENTS.md` は
  定数係数を検証として認めていないので、**A.4.3 の穴は埋まらない**。
  埋まらないものを埋まったように見せないため、本節は走らせていない。
- **H100 / A100 の p=767。**本節は GB200 のみ。§10.2 は H100 では
  emulation が全次数で負けると測っているので、「勝つ次数の精度」という
  本節の問い自体が H100 には無い。
