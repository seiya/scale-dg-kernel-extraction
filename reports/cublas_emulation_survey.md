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
