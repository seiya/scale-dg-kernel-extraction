# SCALE-DG GPU 対応・最適化セッション報告

- 日付: 2026-08-24〜2026-08-25
- 対象リポジトリ: `scale-dg-kernel-extraction`
- 執筆時のブランチ / HEAD: `feature/cuda` / `299a868`
- 対象GPU: NVIDIA GB200（RIKYU）

## 1. 目的と到達点

本セッションでは、元の OpenMP/CPU 向け DG advection kernel extraction に
対して、次の順に GPU 実装と性能調査を進めた。

1. OpenACC 対応を追加し、元の構造を残した `OPENACC_ASIS` を比較基準にした。
2. tendency 計算を主要処理単位に分けた `OPENACC_SPLIT` を追加した。
3. 同じ分割を CUDA Fortran で実装した `CUDAFORTRAN_SPLIT` を追加した。
4. p=7 の tendency 全体をまとめる `CUDAFORTRAN_FUSED` を追加・最適化した。
5. p=255、Ne=1 を主対象としつつ Ne>1 にも対応する tiled kernel を追加した。
6. FP64 Tensor Core、cuBLAS GEMM、CUTLASS GEMM、z-GEMM epilogue fusion を順に試した。
7. `nsys` と `ncu` により時間内訳、FLOP、DRAM traffic、occupancy、stall を調査した。
8. 途中で導入された「一定速度・一様 Cartesian を代表スカラーへ縮約する」誤りを
   発見し、元の配列ベースの数値的意味へ戻した。

現在は、p=7 では pointwise field を維持した CUDA FUSED が最速、p=255 では
volume flux を materialize した GEMM と z epilogue assembly fusion が有効、という
構成になっている。

## 2. 最も重要な数値契約

このコードの入力例では速度や幾何係数に同じ値が入るが、実装上はそれらが
点ごと・要素ごとに異なることを許す。最適化時には以下を必ず維持する。

- `q`, `u`, `v`, `w` は pointwise field である。
- `Escale` は点・要素・方向ごとの配列である。
- `normal_fn`, `Fscale` は面点・要素ごとの配列である。
- 体積項は `D(q*u)`, `D(q*v)`, `D(q*w)` であり、一般には
  `u*D(q)`, `v*D(q)`, `w*D(q)` ではない。
- 数値流束は `VMapM/VMapP` の M/P 両側の値と `normal_fn/Fscale` を使って
  6面すべてで評価する。
- 現在の入力で3面が流出側になってゼロでも、その条件を kernel の一般仕様に
  埋め込んではならない。

この契約はルートの `AGENTS.md` にも記録した。

## 3. 問題サイズと scaling の解釈

`mesh_setup` に渡す物理領域は常に `1 x 1 x 1` であり、

```text
dx = 1 / NeX, dy = 1 / NeY, dz = 1 / NeZ
```

である。したがって `NeX/NeY/NeZ` を増やすと計算領域が広がるのではなく、
同じ領域内で要素が細かくなる。通常の weak scaling のような領域拡大ではない。

代表的な比較は次のとおり。

| 条件 | volume DOF | 備考 |
|---|---:|---|
| NeX=NeY=NeZ=32, p=7 (`Nq=8`) | `32^3 * 8^3 = 256^3` | 要素数が多く、各要素は小さい |
| NeX=NeY=NeZ=1, p=193 (`Nq=194`) | `194^3` | 上記とはDOF数も異なる |
| NeX=NeY=NeZ=1, p=255 (`Nq=256`) | `256^3` | volume DOF数だけは p=7/32^3 と同じ |

p=7/32^3 と p=255/1 は DOF 数が同じでも、同じ GPU 問題ではない。行列次数、
面点数、要素間並列性、kernel launch 数、shared-memory reuse、GEMM への写像が
大きく異なる。

性能比較で kernel 実装だけを比較するときは、`NeX/NeY/NeZ`, `PolyOrder`,
`dt`, `nstep`, `output_interval` を変えない。セッション中に benchmark input の
`nstep` と `dt` が元値から変わったことがあり、以後は入力差分を先に確認する
方針とした。

## 4. 実装の変遷

### 4.1 OpenACC

`03fc9e9` で OpenACC 対応を追加した。field、DG operator、mesh map、geometry を
time-stepping 中 GPU resident にし、出力時は min/max scalar reduction のみを
host に戻す構造にした。

`OPENACC_ASIS` は元の `cal_dqdt` の構造を残した基準実装である。

`5142b04` で `OPENACC_SPLIT` を追加した。単に `cal_dqdt` の外側だけを分けるの
ではなく、`divlike_dirxyz` や lift の最下層まで全要素の `ke` loop を含む
all-element kernel を用意し、次の主要 kernel に分けた。

```text
volume flux
tensor-product derivative
surface lift
dqdt assembly
```

初期化直後の重複した halo update は削除し、GPU data region 内で velocity halo を
初期化し、各 RK stage で変化する `q` halo を更新する構造に整理した。

### 4.2 CUDA Fortran SPLIT

`0355529` で `CUDAFORTRAN_SPLIT` と最初の p=7 FUSED を追加した。OpenACC resident
配列は `host_data use_device` で CUDA Fortran に渡し、host 経由の field copy は
行わない。

最初の CUDA SPLIT は OpenACC より tensor-product derivative が少し速くなった
一方、volume flux、lift、assembly などが少しずつ遅く、合計では OpenACC に
勝てないケースがあった。`do while (idx <= npoint)` の grid-stride loop を
1 thread/1 point にする案も試したが、loop の有無だけでは全体の律速は解消せず、
launch 構造と中間配列 traffic の削減が必要と判断した。

### 4.3 p=7 FUSED

p=7 では1要素512点を1 blockで扱い、256 thread が各2点を担当する kernel を
中心にした。`D1D`、3方向の pointwise volume flux、6面の数値流束を shared
memory に置き、微分、lift、assembly を1 kernel内で行う。

主な知見は以下。

- kernel を分けるだけでは intermediate field の read/write と launch overhead が残る。
- p=7 は要素内データが小さく、1 block/element の fusion と shared-memory reuse が効く。
- lift の shared-memory 配置では FP64 bank conflict が問題になり、read-only/global
  cache を使う方が速い場合があった。
- occupancy を上げるだけでは速くならず、register pressure、bank conflict、再利用量を
  同時に見る必要がある。

### 4.4 p=255 FUSED

p=255 (`Nq=256`, `Np=256^3`) は1要素を1 blockにできないため、16x16 tile に分けた。
各要素につき boundary kernel と x/y/z directional kernel を起動する。

- block index に要素番号を含め、Ne=1だけでなく Ne>1 に対応した。
- 巨大な dense `Lift_mat(256,256,256,6)` は保持せず、separable な
  `Lift1D(256,6)` を使う。
- p=255 の LGL node、`D1D`、`Lift1D` は起動時に生成する。
- boundary temporary は6面分を保持する。
- 各 directional kernel は `D(q*velocity)`、対応する2面の lift、pointwise
  `Escale` を評価する。

## 5. 代表スカラー特殊化の誤りと修正

旧 `93a758f` では p=7 FUSED の高速化時に、

```fortran
adv_x = Escale(1,1,1) * u(1,1)
adv_y = Escale(1,1,2) * v(1,1)
adv_z = Escale(1,1,3) * w(1,1)
```

のような代表スカラーを kernel に渡し、`D(q*u)` を `adv*D(q)` に置き換えた。
さらに Cartesian face と速度符号から3つの inflow face だけを計算した。

これは benchmark input の現在値には一致しても、元の配列 API が想定する問題を
狭めており、不正な最適化である。調査の結果、`0355529` の時点では配列を参照して
おり、問題は `93a758f` で初めて入ったことを確認した。

修正内容:

- p=7/p=255 の両方へ `u/v/w`, `Escale`, `VMapM/P`, `normal_fn`, `Fscale` を渡す。
- volume flux を点ごとに `q*u`, `q*v`, `q*w` として作る。
- 元式の Rusanov/upwind numerical flux を6面すべてで評価する。
- lift も6面すべてを加算する。
- p=255 face temporary を3面分から6面分へ変更する。

点ごとに速度と幾何係数を意図的に変えた p=7 テストで、
`CUDAFORTRAN_FUSED` と `CUDAFORTRAN_SPLIT` の owned `dqdt` 全点を比較し、

```text
max_abs_diff = 0.0
```

を確認した。p=255/Ne=1 も100 stepを完走し、修正直後の device event は
300 tendency call 合計約1.494 s、約4.98 ms/callだった。

この修正は旧 `93a758f` を amend し、次のコミットになった。

```text
03551c7 Optimize CUDA Fortran fused tendency kernels
```

重要: 代表スカラー版で取得した「compute 11%、memory throughput 50%」などの
NCU値や FLOP/byte 算定は、現行の array-correct kernel と同じ仕事量ではない。
最適化のヒントにはなったが、現行版の効率値として再利用してはならない。

## 6. profiling の進め方と得た知見

### 6.1 実行方法

RIKYU では通常の GPU executable は login node で実行できる。一方、`nsys` と
`ncu` は login node で直接起動せず、`sbatch` で GPU job として実行する。

OpenACC/CUDA build と Slurm profiling job の双方で、環境に応じて次を先に行う。

```bash
module load nvhpc
# または
module load nvhpc-hpcx
```

GB200 向け build:

```bash
make clean
make CUDA=1 GPUFLAGS=-gpu=cc100
```

build mode を変えると `.o` と `.mod` は互換でないため、必ず `make clean` する。

### 6.2 nsys

`nsys` は次の用途に使った。

- kernel launch の時系列と回数
- tendency 以外の `q -> q0`、RK update、halo update、min/max reduction の比率
- host synchronization と launch overhead
- FUSED による intermediate kernel 消滅の確認

`slurm-40067.out` などを起点に解析した。tendency を高速化すると、以前は小さかった
`q -> q0`、RK update、halo update の比率が相対的に目立つようになる。

### 6.3 ncu

最初の `ncu` 実行は `slurm-40068.out` のように失敗したため、profiling 専用の
Slurm scriptを作り、kernel regex、`--launch-skip`、`--launch-count` を指定して
主要 kernel を1つずつ採取する方式にした。full set は非常に重いため、最初は
必要 metric のみに絞り、必要な kernelだけ full/source view を取る方がよい。

FLOP と DRAM bandwidth は原則として次で集計した。

```text
FP64 FLOP = DADD + DMUL + 2 * DFMA
DRAM bandwidth = (dram read bytes + dram write bytes) / kernel duration
```

理論値は計算式から求めた数学的仕事量、NCU値は実際に発行された instruction と
DRAM transaction から求めた値として分けて表示する。cache hit、tile間の再読、
compiler の FMA 化により両者は一致しない。

GB200 の比較基準として、このセッションでは RIKYU system document に基づき
FP64 40.1 TFLOP/s、HBM3e 7.9 TB/s を用いた。Tensor Core path も同じ 40.1 TFLOP/s を
分母にしてよい。GB200（Blackwell）では Hopper にあった FP64 Tensor Core の 2×
アドバンテージが撤廃されており、**FP64 Tensor Core peak = FP64 CUDA-core peak**
（2 FLOP × 64 FMA/clk/SM × 152 SM × 2.062 GHz = 40.12 TFLOP/s）だからである。
詳細は `overall_summary_report.md` §7 を参照。

## 7. 理論仕事量の目安

以下は array-correct な数学的演算を、multiply/addを各1 FLOP、FMAを2 FLOPとして
数えた概算である。`abs` と符号反転は FLOP に含めない。numerical flux は1面点
あたり約20 FLOPとした。

### 7.1 p=7, Ne=32^3

```text
Nq     = 8
Np*Ne  = 512 * 32768 = 16,777,216 points
Nface  = 6 * 8^2 * 32768 = 12,582,912 face points
```

| 処理 | FLOP / tendency（概算） |
|---|---:|
| pointwise volume flux `q*u/v/w` | 50,331,648 |
| 3方向 tensor derivative | 805,306,368 |
| 6面 numerical flux | 251,658,240 |
| 6面 lift | 184,549,376 |
| pointwise assembly | 100,663,296 |
| 合計 | **1,392,508,928** |

後述の現行 FUSED device time 1.15029 s / 3000 calls を使うと、数学的仕事量ベースで
約3.63 TFLOP/s、40.1 TFLOP/s に対して約9.1%である。これはNCU instruction count
そのものではなく、アルゴリズム上の有効 FLOP/s である。

### 7.2 p=255, Ne=1

```text
Nq     = 256
Np*Ne  = 256^3 = 16,777,216 points
Nface  = 6 * 256^2 = 393,216 face points
```

| 処理 | FLOP / tendency（概算） |
|---|---:|
| pointwise volume flux `q*u/v/w` | 50,331,648 |
| 3方向 tensor derivative | 25,769,803,776 |
| 6面 numerical flux | 7,864,320 |
| 6面 lift | 184,549,376 |
| pointwise assembly | 100,663,296 |
| 合計 | **26,113,212,416** |

通常 FUSED の device time 14.9659 s / 3000 calls では約5.24 TFLOP/s相当である。
ただし tiled FUSED は同じ `q*velocity` をtile間で再計算するため、実発行 FLOP は
この数学的最小値より多い。GEMM/Tensor Core path の効率を論じる場合は NCU の
Tensor instruction を別途使うが、peak は CUDA-core と同じ 40.1 TFLOP/s でよい
（GB200 では両者が同値。§6 の注記を参照）。

## 8. `q -> q0` copy と memory traffic

p=7/32^3 と p=255/1 はどちらも owned volume point が `256^3` 個なので、FP64の
`q -> q0` は1 stepあたり次の最低 trafficを持つ。

```text
read  = 256^3 * 8 byte = 134.2 MB
write = 256^3 * 8 byte = 134.2 MB
total = 268.4 MB / step
```

この kernel は演算をほぼ持たない pure streaming kernel である。tendency fusion後に
相対比率が上がったため調査対象になったが、SSP-RKで step開始時の `q0` を保持する
意味は必要であり、単純削除はできない。さらに最適化する場合は、alignment、vector
load/store、実DRAM byte、L2 hit、他kernelとのfusion可能性を確認する。ただし RK の
意味や `q0` lifetime を変えてはならない。

## 9. 現行実装の統一条件での時間比較

`execution_times.md` / `bench_runs/slurm-41348.out` に保存された測定を要約する。

条件:

- NVIDIA GB200 1 GPU、host `c162`
- commit系列: `69ef5e6` 時点の主要path比較
- `nstep=1000`, `DGOptrKernel_OptType=OPT1`
- p=7: Ne=32^3, dt=1e-5
- p=255: Ne=1, dt=1e-7
- CUDA build: `make CUDA=1 GPUFLAGS=-gpu=cc100`

### 9.1 p=7, Ne=32^3

| path | Main [s] | Cal_tend [s] | volume+lift wall [s] | CUDA device [s] |
|---|---:|---:|---:|---:|
| OPENACC_ASIS | 3.634 | 3.113 | 2.529 | — |
| OPENACC_SPLIT | 2.901 | 2.390 | 1.804 | — |
| CUDAFORTRAN_SPLIT | 2.868 | 2.351 | 1.765 | flux 0.465 / deriv 0.580 / lift 0.213 / assembly 0.472 |
| CUDAFORTRAN_FUSED | **1.695** | **1.182** | **1.182** | fused **1.150** |
| CUDAFORTRAN_FUSED_TC | 2.019 | 1.506 | 1.505 | fused 1.473 |
| CUDAFORTRAN_GEMM | 9.271 | 8.743 | 8.742 | GEMM 8.705 |

結論:

- p=7 は小さい要素行列が多数あるため、単純な GEMM 化は不利。
- CUDA FUSED が OPENACC_ASIS の Main に対して約2.14倍高速。
- p=7 の FP64 Tensor Core版は通常 FUSED より遅い。ただしこの遅さの原因は
  MMA 命令ではなく shared memory のバンクコンフリクトだった。commit `e22dda1`
  でレイアウトを組み替えた結果、**p=7 の最速パスは `CUDAFORTRAN_FUSED_TC` に
  入れ替わった**（同一条件で Main 1.614 s / CUDA device 1.068 s、job 43618）。
  上表は `299a868` 時点の値である。詳細は
  `tc_paper_survey_2407.09621.md` §5-6。

### 9.2 p=255, Ne=1

| path | Main [s] | Cal_tend [s] | volume+lift wall [s] | CUDA device [s] |
|---|---:|---:|---:|---:|
| CUDAFORTRAN_FUSED | 15.529 | 15.000 | 14.999 | 14.966 |
| CUDAFORTRAN_FUSED_TC | 13.760 | 13.237 | 13.236 | 13.204 |
| CUDAFORTRAN_GEMM | **4.439** | **3.906** | **3.905** | **3.866** |

結論:

- p=255 は大きな dense contraction になり、GEMM 化が有効。
- GEMM は通常 FUSED の Main に対して約3.50倍高速。
- 同じ総DOFでも p=7/32^3 と最適戦略が逆になる。
- cuBLAS floating-point emulation API は導入toolkitで利用できず、native FP64へ
  fallback後に timeoutした試行がある。

## 10. Tensor Core / CUTLASS / fusion の追加調査

`69ef5e6` で FP64 MMA (`m8n8k4`) と cuBLAS GEMM pathを追加し、`299a868` で
CUTLASS d884 GEMM と z-epilogue assembly fusion を追加した。詳細な生データと
試行は `p255_gemm_fusion_session_report.md` に保存されている。

### 10.1 `q*velocity` を GEMM mainloopへ融合する試行

p=255、nstep=1000 の比較:

| 構造 | device-event |
|---|---:|
| volume_flux + cuBLAS + assembly | 約3.87 s |
| volume_flux + CUTLASS d884 + assembly | 約3.91 s |
| mainloop内で `q*velocity` を生成 | 約6.48 s |

mainloop fusion は約1.66倍遅かった。消せる volume-flux kernel は約0.45〜0.5 s
程度なのに対し、CTAごとの `q/velocity` 再読と multiply、標準 global-to-shared
pipeline の崩れで2.5 s以上を失った。

したがって p=255 では volume flux array は単なる無駄な中間配列ではなく、
高効率 GEMM のための materialized input / cacheable preprocessing と考えるべきである。

### 10.2 z GEMM epilogueへの assembly fusion

採用した構造:

```text
volume_flux
-> CUTLASS x GEMM -> deriv_x
-> CUTLASS y GEMM -> deriv_y
-> lift
-> CUTLASS z GEMM + dqdt assembly epilogue
```

`SCALE_DG_VARYING_COEFF=1` の point-varying testで cuBLAS GEMM pathとの
`dqdt` maxabsは `3.55e-15`。

| path | device-event | volume GEMM |
|---|---:|---:|
| cuBLAS | 3.881 s | — |
| CUTLASS CUTE | 3.914 s | 2.284 s |
| z-epilogue FUSED | **3.603 s** | 2.444 s |

z epilogueは追加loadにより GEMM部分を約0.16 s遅くするが、独立 assembly kernelを
消す効果が上回り、CUTE比約8%、cuBLAS比約7%改善した。

### 10.3 不採用にした epilogue 微修正

| 試行 | 数値 | 性能 | 判断 |
|---|---|---|---|
| shared-memory barrierをまとめる | maxabs約500 | 改善なし | tile対応を壊すため不採用 |
| auxiliary fragmentの寿命短縮 | maxabs約2e-15 | 約3.70 sへ悪化 | load直列化が不利 |
| epilogue loopをfull unroll | maxabs約3.6e-15 | 約3.69 sへ悪化 | register/code-size面で不利 |

現在残しているのは、volume flux materialization、通常の Tensor Core mainloop、
iterationごとの同期、複数operandのまとめ読み、`unroll(1)` である。

## 11. 失敗・注意事項から得た一般則

1. **現在の入力値と API の意味を混同しない。** 定数が入っていても配列は配列として扱う。
2. **高速化前に数値契約をテストで固定する。** constant inputだけでは scalarization bugを検出できない。
3. **同じDOF数は同じGPU問題を意味しない。** polynomial orderとelement countで並列構造が変わる。
4. **kernel数削減は常に正義ではない。** GEMM mainloopを壊すfusionは中間配列削減より高くつく。
5. **occupancy単独で判断しない。** register、shared bank conflict、cache reuse、instruction mixを見る。
6. **理論FLOP/byteとNCU実測を分ける。** tile再読やcacheにより意味が違う。
7. **device-eventとwall timeを混ぜない。** launch/syncを含むかを表に明記する。
8. **profilingは同一input・同一commitで行う。** 古い scalar-specialized profileは現行版に適用しない。
9. **最適化は速くなったものだけ残す。** 数値不一致は速度に関係なく即revertする。

## 12. コミットと成果物

主要コミット:

| commit | 内容 |
|---|---|
| `03fc9e9` | OpenACC GPU support |
| `5142b04` | bottom-level OpenACC split kernels |
| `0355529` | CUDA Fortran split / initial fused kernels |
| `03551c7` | array semanticsを維持したFUSED最適化とp=255対応 |
| `69ef5e6` | FP64 Tensor Core / cuBLAS GEMM paths |
| `299a868` | CUTLASS paths / p=255 z-epilogue assembly fusion |
| `e22dda1` | p=7 Tensor Core kernel の shared memory レイアウト刷新（バンクコンフリクト除去） |

関連ファイル:

- `execution_times.md`: 同一条件での path 別実行時間
- `p255_gemm_fusion_session_report.md`: p=255 GEMM/fusion の詳細
- `tc_paper_survey_2407.09621.md`: arXiv:2407.09621 の調査と p=7 TC カーネル刷新
- `bench_runs/`: 比較用 input、job、ログ
- `output/`: nsys/ncu report群
- `README.md`: 現行実装と実行方法
- `AGENTS.md`: 数値契約、build、validation、profiling、commit方針

job shell、Slurm output、Nsight report は、ユーザーが明示的に指定しない限り
source commit には含めない。本レポートを含む Markdown レポート類は、ユーザーの
指示により `reports/` にまとめてコミットしてある。

## 13. 次に行う価値がある調査

- 現行 array-correct p=7 FUSEDを改めて NCU採取し、旧scalar版ではない
  FLOP/s、DRAM bandwidth、stall、register数を確定する。
- p=255 z epilogueについて、mainloopを変えずに liftとのoverlapや独立kernel削減を試す。
- `q -> q0`、RK update、halo updateを streaming kernelとして個別に測定し、
  tendency高速化後の Amdahl limitを定量化する。
- p=7 と p=255 の間の polynomial orderについて、element parallelismとGEMM化の
  crossover pointを測る。
- すべての新pathで point-varying coefficient regressionを自動化する。

ただし最優先は、性能ではなく元の `D(q*velocity)` と6面 numerical fluxの意味を
守り続けることである。
