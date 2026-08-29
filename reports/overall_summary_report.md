# SCALE-DG kernel extraction — GPU 実装・性能 総合まとめレポート

作成日: 2026-08-25
対象リポジトリ: `scale-dg-kernel-extraction`
ブランチ / HEAD: `feature/cuda` / `299a868`
対象GPU: NVIDIA GB200 (RIKYU) 1 GPU
ビルド: `make CUDA=1 GPUFLAGS=-gpu=cc100`（nvcc は `-arch=sm_100`、NVIDIA HPC SDK `nvhpc-hpcx`）

本レポートは、ソースコード、`README.md` / `AGENTS.md`、既存の
`execution_times.md` / `gpu_optimization_session_report.md` /
`p255_gemm_fusion_session_report.md`、および `output/` 配下の
nsys / ncu 測定結果（Slurm job `43219`、`job_all.sh`）を横断して
まとめたものである。

---

## 1. 要約

1. **p=7（Ne=32³）では `CUDAFORTRAN_FUSED` が最速**。tendency の GPU 時間は
   ASIS 比 **2.66×**、SPLIT 比 **1.95×**。単一カーネル内の shared-memory 再利用が効く。
   （2026-08-26 追記: この順位は `e22dda1` 以降 `CUDAFORTRAN_FUSED_TC` に
   入れ替わっている。項目 4 と §7 の現行列を参照。）
2. **p=255（Ne=1）では GEMM 化が最速**。`CUDAFORTRAN_GEMM_FUSED` は
   tiled FUSED 比 **4.30×**、`CUDAFORTRAN_GEMM` 比 **1.08×**。
3. **同じ体積 DOF 数（256³）でも、最適戦略は p=7 と p=255 で逆転する。**
   p=7 は「小行列 × 大量要素」、p=255 は「巨大 dense contraction」で、
   GEMM 化は p=7 では **7.4× 遅く**、p=255 では **4.3× 速い**。
4. **FP64 Tensor Core（`m8n8k4` 直書き）の勝敗は shared memory レイアウトで決まる。**
   本レポート時点（`299a868`）では p=7 で 1.28× 遅く、p=255 でも 1.13× しか
   速くならなかったが、p=7 の原因は MMA 命令そのものではなく shared memory の
   バンクコンフリクトだった。commit `e22dda1` でレイアウトを組み替えた結果、
   **p=7 では Tensor Core 版が最速に入れ替わった**
   （カーネル 662 µs → 497 µs、CUDA core 版比 1.11×）。§4.1 / §5 の p=7
   `CUDAFORTRAN_FUSED_TC` の行はいずれも `299a868` 時点の値である。
   詳細は `tc_paper_survey_2407.09621.md` §5-6。
   なお p=255 の手書き TC 経路は依然として CUTLASS/cuBLAS の multistage
   mainloop に大きく負けており、この結論は変わらない。
5. **カーネル削減は「消せる DRAM トラフィック」の範囲でのみ得。** p=255 で
   assembly（DRAM 92% で帯域律速）を z-GEMM epilogue に融合すると **−8.3%** だが、
   `q*vel` を GEMM mainloop に融合すると +66% と大きく悪化する。
   CUTE と FUSED の同一タイル比較（§6）で、前者は z GEMM の SM throughput を
   87.2% → 79.8% に下げるだけで mainloop 自体は保たれることを確認した。
6. **tendency を速くした結果、tendency 以外が無視できなくなった。** p=7 FUSED では
   GPU 時間の **約 27%** が `q0←q` コピー・RK 更新・halo 更新である（p=255 GEMM_FUSED では約 11%）。
   さらに、カーネルの間で GPU が空く時間も 139 µs/step あった。`q0←q` の融合（§8.1）と
   OpenACC 領域の async 化・ストリーム統一（§8.2）で、p=7 `FUSED_TC` の Main は
   1.4146 → 1.2072 s になった。
7. `output/` の測定は `nstep=10` のため、**cuBLAS/CUTLASS 経路のアプリタイマは
   ライブラリ初期化の一回性コストを含む**。定常性能は nsys のカーネル総和、
   および `nstep=1000` の `execution_times.md` を見ること。

---

## 2. 数値契約（最適化で壊してはならない前提）

`AGENTS.md` に記録済みの内容の再掲。本リポジトリの最適化はすべてこれを満たす。

- `q`, `u`, `v`, `w` は **pointwise field**。ベンチ入力が定数でも代表スカラー化しない。
- `Escale` は点・要素・方向ごと、`normal_fn` / `Fscale` は面点・要素ごとの配列。
- 体積項は `D(q*u)`, `D(q*v)`, `D(q*w)`。一般に `u*D(q)` ではない。
- 数値流束は `VMapM`/`VMapP` の M/P 両側値を使い **6 面すべて**で評価する。
  現在の入力で 3 面が実質ゼロでも、それを仕様に埋め込まない。
- halo は velocity を time-step 前に初期化し、`q` は各 RK stage で更新する。

セッション中、旧 `93a758f` でこの契約を破る「代表スカラー特殊化」が混入し、
`03551c7` で修正された。**その時期に採取した NCU 値・FLOP/byte 見積りは
現行 array-correct 実装の値として使ってはならない。**

検証は点ごとに `u,v,w,Escale,normal_fn,Fscale` を変えた
`SCALE_DG_VARYING_COEFF=1` 回帰で行い、owned `dqdt(:,1:Ne)` の全点比較で
- p=7 FUSED vs SPLIT: `max_abs_diff = 0.0`
- p=255 GEMM_FUSED vs GEMM: `max_abs_diff = 3.55e-15`（丸め誤差レベル）

---

## 3. 実装パス一覧

| `DqdtKernel_Type` | 実装 | p=7 | p=255 |
|---|---|:--:|:--:|
| `OPENACC_ASIS` | 元構造のまま単一 `cal_dqdt` accelerator region | ✓ | — |
| `OPENACC_SPLIT` | flux / derivative / lift / assembly を最下層まで全要素カーネル化 | ✓ | — |
| `CUDAFORTRAN_SPLIT` | 同じ分割を CUDA Fortran で実装 | ✓ | — |
| `CUDAFORTRAN_FUSED` | p=7: 1 block/element 融合。p=255: 16×16 tile の x/y/z カーネル | ✓ | ✓ |
| `CUDAFORTRAN_FUSED_TC` | 同構造を FP64 Tensor Core `mma.sync.m8n8k4` で実装 | ✓ | ✓ |
| `CUDAFORTRAN_GEMM` | volume flux と数値流束はカーネル、微分と `Lift1D` を cuBLAS Dgemm | ✓ | ✓ |
| `CUDAFORTRAN_GEMM_CUTE` | volume 3 GEMM のみ CUTLASS d884 に置換 | — | ✓ |
| `CUDAFORTRAN_GEMM_FUSED` | 上に加え z GEMM の epilogue で dqdt assembly を融合 | — | ✓ |

OpenACC の resident 配列は `host_data use_device` で CUDA Fortran に渡し、
time-stepping 中に host 経由の field copy は行わない。
p=255 は dense `Lift_mat(256,256,256,6)` を持たず separable な `Lift1D(256,6)` を
起動時に生成する。

---

## 4. 実行時間の比較（`nstep=1000`, `execution_times.md`, Slurm job 41348）

`DGOptrKernel_OptType=OPT1`、p=7 は Ne=32³/dt=1e-5、p=255 は Ne=1/dt=1e-7。
`CUDA device` は CUDA Event による device 時間（host launch/sync を含まない）。
`volume+lift` は end-to-end wall time。

### 4.1 p=7, Ne=32³

| path | Main [s] | Cal_tend [s] | volume+lift wall [s] | CUDA device [s] |
|---|---:|---:|---:|---:|
| `OPENACC_ASIS` | 3.634 | 3.113 | 2.529 | — |
| `OPENACC_SPLIT` | 2.901 | 2.390 | 1.804 | — |
| `CUDAFORTRAN_SPLIT` | 2.868 | 2.351 | 1.765 | flux 0.465 / deriv 0.580 / lift 0.213 / asm 0.472 |
| **`CUDAFORTRAN_FUSED`** | **1.695** | **1.182** | **1.182** | **fused 1.150** |
| `CUDAFORTRAN_FUSED_TC` | 2.019 | 1.506 | 1.505 | fused 1.473 |
| `CUDAFORTRAN_GEMM` | 9.271 | 8.743 | 8.742 | GEMM 8.705 |

Main 時間比: FUSED は ASIS の **2.14×**、GEMM は **0.39×**。

### 4.2 p=255, Ne=1

| path | Main [s] | Cal_tend [s] | volume+lift wall [s] | CUDA device [s] |
|---|---:|---:|---:|---:|
| `CUDAFORTRAN_FUSED` | 15.529 | 15.000 | 14.999 | 14.966 |
| `CUDAFORTRAN_FUSED_TC` | 13.760 | 13.237 | 13.236 | 13.204 |
| **`CUDAFORTRAN_GEMM`** | **4.439** | **3.906** | **3.905** | **3.866** |

同条件の別セッション実測（device-event, nstep=1000）:
cuBLAS **3.881 s** / CUTE **3.914 s** / **z-epilogue FUSED 3.603 s**
（CUTE 比 −8%、cuBLAS 比 −7%）。

---

## 5. nsys によるカーネル内訳（`nstep=10`, job 43219）

`nstep=10` × RK 3 stage = **tendency 30 回**。以下は nsys `cuda_gpu_kern_sum`
の総和を 30 で割った **1 tendency 呼び出しあたりの GPU カーネル時間**。
アプリタイマと違い、launch gap とライブラリ初期化を含まない。

| path | tendency [µs/call] | カーネル種類 | launch/call | 対 FUSED(p7) / 対 GEMM_FUSED(p255) |
|---|---:|---:|---:|---:|
| `OPENACC_ASIS` p7 | 1008.6 | 2 | 2 | 2.66× 遅 |
| `OPENACC_SPLIT` p7 | 730.8 | 5 | 5 | 1.93× 遅 |
| `CUDAFORTRAN_SPLIT` p7 | 738.4 | 5 | 5 | 1.95× 遅 |
| **`CUDAFORTRAN_FUSED` p7** | **379.5** | 1 | 1 | **1.00×** |
| `CUDAFORTRAN_FUSED_TC` p7 | 486.3 | 1 | 1 | 1.28× 遅（`e22dda1` で逆転、§1-4 参照） |
| `CUDAFORTRAN_GEMM` p7 | 2826.6 | 12 | 24 | 7.45× 遅 |
| `CUDAFORTRAN_FUSED` p255 | 4967.9 | 4 | 4 | 4.30× 遅 |
| `CUDAFORTRAN_FUSED_TC` p255 | 4416.4 | 4 | 4 | 3.83× 遅 |
| `CUDAFORTRAN_GEMM` p255 | 1249.7 | 11 | 12 | 1.08× 遅 |
| `CUDAFORTRAN_GEMM_CUTE` p255 | 1259.8 | 12 | 12 | 1.09× 遅 |
| **`CUDAFORTRAN_GEMM_FUSED` p255** | **1154.6** | 11 | 11 | **1.00×** |

観察:

- **OpenACC SPLIT と CUDA Fortran SPLIT はほぼ同じ**（730.8 vs 738.4 µs）。
  「分割したまま」では言語を変えても速くならない。効いたのは **融合**である。
- p=7 GEMM が壊滅的なのは、8×8 の小行列 GEMM を 1 tendency あたり 24 回
  launch する構造になるため。
- p=255 GEMM_FUSED は GEMM 比 −95 µs/call。内訳は下記のとおり
  assembly カーネル（−152.8 µs）を消し、z GEMM が +18 µs 重くなった差分に一致する。

### 5.1 p=255 GEMM 系の内訳（µs / tendency call）

| kernel 群 | `CUDAFORTRAN_GEMM`（cuBLAS） | `CUDAFORTRAN_GEMM_CUTE` | `CUDAFORTRAN_GEMM_FUSED` |
|---|---:|---:|---:|
| GEMM 群合計（volume x/y/z + lift） | 917.0 | 926.1 | 974.4 |
| ├ volume x/y/z（cuBLAS / 自前 CUTLASS） | 917.0※ | 750.0 | 798.4 |
| │　├ x GEMM `Gemm<64,128,16>` | — | 250.2 | 249.2 |
| │　├ y GEMM `GemmBatched<64,64,16>` | — | 244.5 | 242.8 |
| │　└ z GEMM `<64,32,16>` | — | 255.2 | **306.4**（assembly epilogue 込み） |
| └ lift（cuBLAS `d884gemm_*`） | 917.0※ | 176.1 | 176.0 |
| `dqdt_assembly_kernel` | 152.8 | 153.3 | **0（epilogue へ融合）** |
| `volume_flux_kernel` | 150.0 | 150.5 | 150.1 |
| `elembnd_flux_kernel` | 20.3 | 20.2 | 20.2 |
| pack / copy 補助カーネル | 9.7 | 9.7 | 9.8 |
| **合計** | **1249.7** | **1259.8** | **1154.6** |

※ cuBLAS 版は volume と lift の両方が同じ `cutlass_80_tensorop_d884gemm_*`
カーネル群にディスパッチされ、nsys 上で分離できないため合算値のみ示す。

**この表は job 43219（`514853f`）の値である。** その後 lift は z-epilogue に
畳み込まれて行が消え（§8.4 / §8.5）、`volume_flux_kernel` は 150.1 → 125.9 µs に
なった（§8.6）。commit `d7b1853` + §8.6 での `GEMM_FUSED` の現行内訳は
z GEMM 339.1 / x GEMM 249.7 / y GEMM 243.5 / `volume_flux` 125.9 /
`elembnd_flux` 19.6 µs、合計約 978 µs/call である。

読み取れること:

- **CUTE と cuBLAS はほぼ同じ**（1259.8 vs 1249.7 µs、+0.8%）。
  volume GEMM を自前 CUTLASS d884 に置き換えること自体には性能上の意味がない。
- **CUTE → FUSED の差は z GEMM と assembly の交換に完全に帰着する。**
  z GEMM が 255.2 → 306.4 µs（**+51.2**）、assembly カーネルが 153.3 → 0（**−153.3**）で、
  差し引き **−105.2 µs（−8.3%）**。x/y GEMM・lift・flux は 1 µs 以下の差しかない。
- 別セッションの `nstep=1000` 実測（cuBLAS 3.881 s / CUTE 3.914 s / FUSED 3.603 s）と
  比率が一致する（CUTE 比 −8%）。

### 5.2 p=7 の内訳（µs / tendency call）

| `OPENACC_SPLIT` | | `CUDAFORTRAN_SPLIT` | |
|---|---:|---|---:|
| `divlike_dirxyz_all_p7` | 202.4 | `volume_deriv_p7_kernel` | 187.1 |
| `cal_elembnd_flux` | 181.4 | `cal_elembnd_flux`(ACC) | 181.8 |
| `assemble_dqdt` | 152.9 | `dqdt_assembly_kernel` | 152.8 |
| `cal_volume_flux` | 128.0 | `volume_flux_kernel` | 150.3 |
| `matvec_lift_hexahedral_all_p7` | 66.1 | `surface_lift_p7_kernel` | 66.5 |
| **合計** | **730.8** | **合計** | **738.4** |

→ `CUDAFORTRAN_FUSED` はこれら 5 カーネル分（中間配列 7 本の read/write を含む）を
**379.5 µs の単一カーネル**に置き換えている。

---

## 6. ncu による効率分析（`--set basic`, GPU Speed Of Light）

**注意**: ncu 実行中はクロックとシリアライズの影響で duration が nsys より長い
（例: p=7 FUSED カーネル nsys 379.5 µs / ncu 550.9 µs）。**時間は nsys、
効率比率は ncu** を使う。

| kernel | grid | blk | SM% | Mem% | DRAM% | L1% | L2% | reg | smem | occ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p7 `tendency_fused_p7_kernel` | 32768 | 256 | 38.5 | **88.6** | 36.5 | 89.6 | 29.6 | 42 | 15.9 KB | 60.0 |
| p7 `tendency_fused_p7_tc_kernel` | 32768 | 256 | 22.6 | 80.5 | 30.2 | 81.3 | 27.1 | 32 | 28.2 KB | 97.1 |
| p7 `volume_deriv_p7_kernel` | 32768 | 512 | 47.8 | **94.1** | 28.9 | 95.4 | 16.2 | 32 | 13.3 KB | 93.2 |
| p7 `surface_lift_p7_kernel` | 32768 | 512 | 44.4 | 92.4 | 22.9 | 96.3 | 17.9 | 32 | 0 | 81.4 |
| p7 `volume_flux_kernel` | 65536 | 256 | 33.9 | 67.0 | **67.0** | 54.5 | 37.9 | 20 | 0 | 83.9 |
| p7 `dqdt_assembly_kernel` | 65536 | 256 | 39.8 | 92.0 | **92.0** | 45.3 | 63.9 | 22 | 0 | 81.2 |
| p7 `cal_elembnd_flux_407` (ACC) | 98304 | 128 | 40.9 | 76.3 | 61.6 | 77.9 | 53.5 | 54 | 0 | 48.9 |
| p7 `cal_dqdt_openacc_asis_470` | 32768 | 32 | 29.5 | 48.6 | 23.6 | 49.1 | 15.4 | **140** | 0 | **18.4** |
| p255 `tendency_x_p255_kernel` | 65536 | 256 | 48.5 | **98.2** | 2.1 | 98.4 | 8.6 | 32 | 4.1 KB | 98.8 |
| p255 `tendency_z_p255_kernel` | 65536 | 256 | 61.5 | 98.0 | 19.3 | 98.3 | 14.6 | 28 | 4.1 KB | 98.8 |
| p255 `tendency_x_p255_tc_kernel` | 262144 | 32 | 18.2 | **99.1** | 2.6 | 99.6 | 33.6 | 32 | 0 | 49.5 |
| p255 `d884gemm_64x32_16x4_nn`（volume） | 8192 | 128 | **87.5** | 70.2 | 6.3 | 71.9 | 15.9 | 80 | 0 | 24.1 |
| p255 `d884gemm_64x64_16x4_nn`（volume） | 4096 | 128 | **87.0** | 46.4 | 6.1 | 48.2 | 10.4 | 124 | 0 | 18.4 |
| p255 `dqdt_assembly_kernel` | 65536 | 256 | 40.0 | 92.0 | **92.0** | 45.5 | 63.9 | 22 | 0 | 81.4 |
| p255 `volume_flux_kernel` | 65536 | 256 | 33.9 | 66.9 | 66.9 | 54.2 | 37.7 | 20 | 0 | 84.0 |

`CUDAFORTRAN_GEMM_CUTE` / `CUDAFORTRAN_GEMM_FUSED` p=255（Slurm job 43246 / 43241 で追加採取）:

| path | kernel | grid | blk | SM% | Mem% | DRAM% | L1% | L2% | reg | dyn smem | occ% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUTE | z GEMM `GemmBatched<64,32,16>` | 8192 | 64 | **86.9** | 54.6 | 6.2 | 55.7 | 13.6 | 156 | 49.2 KB | 12.2 |
| FUSED | z GEMM `GemmBatchedDqdtAssembly<64,32,16>`（epilogue 込み） | 8192 | 64 | **79.8** | 58.4 | **26.0** | 59.5 | 20.9 | **242** | 49.2 KB | 12.2 |
| CUTE | x GEMM `Gemm<64,128,16>` | 2048 | 128 | 87.8 | 34.3 | 10.1 | 35.9 | 8.0 | 212 | 73.7 KB | 12.1 |
| FUSED | x GEMM `Gemm<64,128,16>` | 2048 | 128 | 88.0 | 34.3 | 10.1 | 36.0 | 8.0 | 212 | 73.7 KB | 12.1 |
| CUTE | y GEMM `GemmBatched<64,64,16>` | 4096 | 128 | 89.2 | 47.8 | 6.3 | 48.9 | 10.2 | 130 | 65.5 KB | 18.4 |
| FUSED | y GEMM `GemmBatched<64,64,16>` | 4096 | 128 | 89.1 | 47.8 | 6.3 | 49.0 | 10.3 | 130 | 65.5 KB | 18.4 |
| CUTE | `dqdt_assembly_kernel` | 65536 | 256 | 39.3 | 92.1 | **92.1** | 44.6 | 63.0 | 22 | 0 | 82.5 |
| 両者 | `d884gemm_64x128_16x3_nt`（lift, cuBLAS） | 2048 | 128 | 27.8–28.0 | 48.3–48.6 | 26.0–26.1 | 51.7–51.8 | 16.5 | 200 | 73.7 KB | 11.8 |
| 両者 | `d884gemm_64x64_16x4_nt`（lift, cuBLAS） | 4096 | 128 | 32.2–32.4 | 67.1–67.3 | 29.7–29.8 | 71.8–71.9 | 18.9–19.0 | 116 | 65.5 KB | 17.5 |
| 両者 | `d884gemm_64x128_16x3_nn`（lift, cuBLAS） | 2048 | 128 | 37.0–37.1 | 56.1–56.3 | 13.0 | 60.9–61.0 | 18.6 | 212 | 73.7 KB | 11.9 |
| 両者 | `volume_flux_kernel` | 65536 | 256 | 33.7–33.9 | 66.1–67.6 | **66.1–67.6** | 54.0–54.1 | 36.6–37.5 | 20 | 0 | 84.0 |
| 両者 | `elembnd_flux_kernel` | 1536 | 256 | 8.8–8.9 | 53.9–62.9 | 53.9–62.9 | 32.1–34.4 | 40.3–47.3 | 32 | 0 | 75.6–80.2 |

読み取れること:

- **手書き tile カーネル（FUSED / FUSED_TC）は L1/TEX throughput で飽和している。**
  （`FUSED_TC` p=7 については、この飽和の実体が shared memory の
  バンクコンフリクトであることが後に判明した。`e22dda1` で除去済み。）
  p=255 で L1 98–99.6%、DRAM わずか 2–3%。データ再利用そのものは非常に良いが、
  SM throughput は 18–62% にとどまる。**律速は演算でも DRAM でもなく shared/L1 帯域**。
- **CUTLASS/cuBLAS の d884 GEMM は SM throughput 87%** に達し、同じ仕事を
  演算律速に持ち込めている。これが p=255 で GEMM が 4.3× 速い理由。
- **`dqdt_assembly_kernel` は DRAM 92% の純帯域律速**。この 152.8 µs は
  「消せば必ず得する DRAM トラフィック」であり、z-epilogue 融合はここを取っている。
- **`cal_dqdt_openacc_asis` は register 140 本 / occupancy 18.4%**。ASIS が遅い理由は
  演算量ではなく、要素ローカル一時配列によるレジスタ圧である。
- **p=7 TC 版は occupancy 97% だが 1.28× 遅い**。occupancy 単独は性能指標にならない。
- FUSED(p=7) は theoretical occupancy 62.5% が **レジスタ数（42）で律速**。ただし
  Mem% 88.6 のため、occupancy を上げても L1 帯域が先に飽和する可能性が高い。
- **z-epilogue 融合の代償が、同一タイル・同一 launch 形状の対照実験で確定した。**
  CUTE と FUSED の z GEMM はどちらも `GemmShape<64,32,16>`、grid 8192、
  block 64 threads、dynamic shared memory 49.2 KB で完全に同一である。
  epilogue を載せると
  **SM throughput 86.9% → 79.8%（−7.1 ポイント）**、
  **DRAM throughput 6.3% → 26.0%**（`Ex*Dx`, `Ey*Dy`, lift の読み込みと `dqdt` 書き出し）、
  **レジスタ 156 → 242 本**となる一方、**occupancy は 12.2% で変わらない**。
- **x/y GEMM と lift、flux は CUTE と FUSED で数値がほぼ完全に一致する**
  （x: 87.8 vs 88.0%、y: 89.2 vs 89.1%）。すなわち **mainloop は壊れていない**。
  これは、以前に不採用とした「`q*vel` を mainloop に融合する案」（1.66× 悪化）との
  決定的な違いである。
- したがって融合の得失は、「z GEMM の SM throughput −7.1 ポイント（+51.2 µs）」と
  「DRAM 92% で帯域律速の assembly カーネル 153.3 µs の消滅」の交換であり、
  §5.1 の −8.3% と完全に整合する。
- なお `dqdt_assembly_kernel` は DRAM 92.1%、`volume_flux_kernel` は 66–68% と
  どちらも帯域律速。**融合で得られる上限は、消せるカーネルの DRAM 時間そのもの**である。
  （`volume_flux_kernel` の「帯域律速」は誤りだった。DRAM 66% で他のどのユニットも
  飽和しておらず、実体はレイテンシ律速である。§8.6 でロードをまとめると
  DRAM 83.4% / 125.9 µs になった。この表の値は変更前のものである。）

---

## 7. 理論仕事量に対する達成効率

FLOP は array-correct な数学的演算数（multiply/add=1、FMA=2、`abs`/符号は除外、
数値流束は 1 面点あたり約 20 FLOP）。**NCU の発行命令数ではない**。
実装（特に tile 化）は同じ `q*vel` を再計算するため、実発行 FLOP はこれより多い。

- p=7, Ne=32³: **1.3925 GFLOP / tendency**
  （flux 0.050 / derivative 0.805 / numerical flux 0.252 / lift 0.185 / assembly 0.101）
- p=255, Ne=1: **26.113 GFLOP / tendency**
  （flux 0.050 / derivative 25.770 / numerical flux 0.008 / lift 0.185 / assembly 0.101）

nsys のカーネル時間で割った **アルゴリズム有効 FP64 FLOP/s**（比は
RIKYU system document の FP64 peak 40.1 TFLOP/s）。

**この表は 2026-08-26 に現行ツリー（`63a4234` + §13.2 の ±x 面ステージング）で
全パスを採り直した**（Slurm job `49700`、`nstep=20`、`nsys profile --trace=cuda`、
`conf_perf_*` と同一条件）。比較のため `514853f` 時点の値を併記する。
p=255 の GEMM 3 経路の µs/call は `elembnd_flux_kernel` を含まない
（§8.7 で 2 本目のストリームに移してあり、x GEMM の裏に完全に収まる）。
それ以外の経路では境界流束もクリティカルパス上なので含めてある。

| path | µs/call（`514853f`）| µs/call（現行）| 有効 TFLOP/s | 対 40.1 TFLOP/s |
|---|---:|---:|---:|---:|
| p7 `OPENACC_ASIS` | 1008.6 | 1011.2 | 1.38 | 3.4% |
| p7 `OPENACC_SPLIT` | 730.8 | 730.6 | 1.91 | 4.8% |
| p7 `CUDAFORTRAN_SPLIT` | 738.4 | 715.7 | 1.95 | 4.9% |
| p7 `CUDAFORTRAN_FUSED` | 379.5 | 323.8 | 4.30 | 10.7% |
| p7 `CUDAFORTRAN_FUSED_TC` | 486.3 | **264.9** | **5.26** | **13.1%** |
| p7 `CUDAFORTRAN_GEMM` | 2826.6 | 1670.7 | 0.83 | 2.1% |
| p255 `CUDAFORTRAN_FUSED` | 4967.9 | 4978.9 | 5.24 | 13.1% |
| p255 `CUDAFORTRAN_FUSED_TC` | 4416.4 | 4420.9 | 5.91 | 14.7% |
| p255 `CUDAFORTRAN_GEMM` | 1249.7 | 1143.0 | 22.85 | 57.0% |
| p255 `CUDAFORTRAN_GEMM_CUTE` | 1259.8 | 1138.1 | 22.95 | 57.2% |
| p255 `CUDAFORTRAN_GEMM_FUSED` | 1154.6 | **970.7** | **26.90** | **67.1%** |

動いた量は経路ごとに大きく違う。**p=7 `FUSED_TC` が 1.84×**
（occupancy・shared レイアウト・±x 面ステージング、`tc_paper_survey` §7〜§13）、
**p=7 `GEMM` が 1.69×**、**p=255 `GEMM_FUSED` が 1.19×**（lift の epilogue 融合と
volume flux のロードまとめ、§8.4〜§8.6）。一方 **p=255 の tiled 経路（`FUSED` /
`FUSED_TC`）と p=7 の OpenACC 経路は ±0** で、この間の最適化がどれも
それらの内側には入っていないことを示している。

そのため順位も 2 か所で入れ替わった。**p=7 の最速は `FUSED` から
`FUSED_TC` へ**（`e22dda1` 以降、§4 項目 4）。p=255 の `GEMM` と `GEMM_CUTE` は
1143.0 対 1138.1 µs で逆転しているが、差 0.4% は run 間のばらつきと同程度なので
両者は同着と見るべきである。最速が `GEMM_FUSED`、p=7 が要素並列カーネル、
という全体の結論は変わらない。

**Tensor Core peak について**: GEMM / TC 経路は FP64 Tensor Core を使うため、
本来は CUDA-core FP64 peak ではなく **FP64 Tensor Core peak** を分母にすべきである。
ただし GB200（Blackwell）では Hopper にあった FP64 Tensor Core の 2× アドバンテージが
撤廃されており、**FP64 Tensor Core peak = FP64 CUDA-core peak = 40.1 TFLOP/s** である。
どちらも 2 FLOP × 64 FMA/clk/SM × 152 SM × 2.062 GHz = 40.12 TFLOP/s で、
RIKYU system document の値と一致する（SM 数は §11、SM clock は `nvidia-smi` の
`clocks.max.sm`）。したがって上表の「対 40.1 TFLOP/s」列は、CUDA-core 経路にも
Tensor Core 経路にも**そのまま適用できる**（分母の差し替えは不要）。
参考として、Hopper 型の 2×（80.2 TFLOP/s）を仮定した場合は上表の値が半分になる。

64 FMA/clk/SM は公表諸元からの導出値であり、実測ではない。厳密な実測 peak が
必要な場合は `cutlass_profiler --operation=Gemm --A=f64 --B=f64 --m=8192 --n=8192
--k=8192` を Slurm 経由で採取すること。

p=255 `CUDAFORTRAN_GEMM_FUSED` の **volume GEMM 単体**を上の分母で見ると
（derivative 25.770 GFLOP = 3 × 2 × 256⁴、方向ごとに 8.590 GFLOP。
z にはさらに numerical flux 0.008 / lift 0.185 / assembly 0.101 が乗る）。
時間は上と同じ job `49692` / `49700`:

| GEMM | µs/call | 有効 TFLOP/s | 対 40.1 TFLOP/s |
|---|---:|---:|---:|
| x `Gemm<64,128,16>` | 254.2 | 33.80 | **84.3%** |
| y `GemmBatched<64,64,16>` | 242.9 | 35.37 | **88.2%** |
| z `<64,32,16>`（assembly + lift epilogue 込み） | 338.6 | 26.23 | 65.4% |
| volume 3 本 合計（derivative のみを分子に） | 835.7 | **30.84** | **76.9%** |

`514853f` 時点の同じ表は x 249.2 / y 242.8 / z 306.4 µs、volume 3 本 798.4 µs
（80.5%）で、そこに独立した lift カーネル 176 µs が続いていた
（「volume + lift 974.4 µs、66.4%」）。§8.4 / §8.5 で lift を z epilogue に
入れたので、**z が 306.4 → 338.6 µs に増える代わりに 176 µs のカーネルが
丸ごと消えた**。差し引き 974.4 → 835.7 µs である。

tendency 全体が 67.1% に留まるのは演算性能ではなく、`volume_flux_kernel`
（129 µs、FLOP 1% だが DRAM 87.5%）のような DRAM 律速カーネルが時間を
占めるためである。

p=7 の効率が低いのは演算能力不足ではない。1 要素あたり 512 点・8×8 行列という
サイズでは算術強度が低く、ncu が示すとおり **L1/shared 帯域**で頭打ちになるため
（現行の `FUSED_TC` で L1/TEX 91%、DRAM 75.6%、FLOP 13.1%）。

---

### 7.1 追記: 現行ツリーでの最速パスの達成効率（2026-08-26 更新）

上の表は `514853f` 時点の値である。その後 §8.4〜§8.8 で volume flux のロード、
lift の epilogue 融合、境界流束の重ね合わせ、RK 更新の 1 次元化が入り、
さらに §13.2 で p=7 の ±x 面 gather が shared 経由になったので、
**各次数の最速パスだけ**を採り直した。

本節は 2026-08-25 に `78fbbf8` で作り、2026-08-26 に現行ツリー
（`63a4234` + §13.2 の ±x 面ステージング）で採り直して**置き換えた**もので
ある。旧版との差は p=7 側だけで、tendency 277.5 → 264.4 µs、
5.02 → 5.27 TFLOP/s、5.75 → 5.98 TB/s、Main 1.131 → 1.066 秒だった。
p=255 側は同じカーネル・同じトラフィックで、差は run 間のばらつきの範囲。

時間は nsys（`nstep=20`、Slurm job `49692`）、DRAM バイトは ncu 実測
（job `49699`。p=255 は job 46493 の値を時間で割り直したもので、
トラフィックは変わっていない）、分母は 40.1 TFLOP/s と 7.9 TB/s。

**p=7, `CUDAFORTRAN_FUSED_TC`, Ne=32³**

| | µs | TFLOP/s | 対 40.1 | TB/s | 対 7.9 |
|---|---:|---:|---:|---:|---:|
| tendency（単一カーネル） | 264.4 | **5.27** | 13.1% | **5.98** | **75.6%** |
| 1 step（GPU カーネル計） | 1033.1 | 4.04 | 10.1% | 6.08 | 76.9% |
| 1 step（wall = Main/1000） | 1065.6 | 3.92 | 9.8% | 5.89 | 74.6% |

**p=255, `CUDAFORTRAN_GEMM_FUSED`, Ne=1**

| | µs | TFLOP/s | 対 40.1 | TB/s | 対 7.9 |
|---|---:|---:|---:|---:|---:|
| tendency（クリティカルパス） | 964.9 | **27.06** | **67.5%** | 2.63 | 33.2% |
| ├ `volume_flux_kernel` | 129.2 | 0.39 | 1.0% | 6.92 | **87.5%** |
| ├ x GEMM `Gemm<64,128,16>` | 254.2 | 33.80 | **84.3%** | 1.45 | 18.4% |
| ├ y GEMM `GemmBatched<64,64,16>` | 242.9 | 35.37 | **88.2%** | 0.93 | 11.8% |
| └ z GEMM + assembly + lift | 338.6 | 26.23 | 65.4% | 2.76 | 34.9% |
| （`elembnd`、2 本目のストリームで隠れる） | 26.9 | 0.30 | 0.7% | 3.94 | 49.9% |
| 1 step（GPU カーネル計） | 3139.2 | 24.95 | 62.2% | 2.91 | 36.9% |
| 1 step（wall = Main/1000） | 3228.0 | 24.27 | 60.5% | 2.84 | 35.9% |

tendency のクリティカルパスは `elembnd_flux_kernel` を含まない（§8.7 で x GEMM の
裏に移した）。1 step は tendency 3 回 + RK 更新（p=7 は stage 1 が 76.7 µs、
stage≥2 が 74.1 µs ×2、p=255 は 77.1 と 74.5 µs ×2）+ halo 3 回である。

読み取れること:

- **同じ 16.78 M 自由度でも性格が正反対である。** p=7 は FLOP 13.1% / DRAM 76%
  / L1 91% で L1・帯域律速、p=255 は FLOP 67.5% / DRAM 33% で演算律速。
  §10-3 の「同じ DOF 数は同じ GPU 問題を意味しない」がそのまま数字に出ている。
- **p=7 の TC 経路は §7 の 2.86 TFLOP/s から 5.27 TFLOP/s になった**（486.3 →
  264.4 µs/call）。占有率と shared memory レイアウトの作業（`tc_paper_survey`
  §7〜§11）と、±x 面 gather のステージング（同 §13）の効果である。
  DRAM は 75.6% まで来ており、**転送量は理論最小の 1.02 倍**なので、
  この経路で残っているのは L1 トランザクション数だけである。
- **p=255 の tendency は 22.62 → 27.06 TFLOP/s（56.4 → 67.5%）。** 67.5% に
  留まっている差分の大半は `volume_flux_kernel` の 129 µs（FLOP 1% だが
  DRAM 87.5% で帯域は使い切っている）と z GEMM の 65.4% で、x/y GEMM を
  88% から上げる余地はほとんど無い。
- **GPU の稼働率は高い。** p=7 で 1033.1 / 1065.6 = 96.9%、p=255 で
  3139.2 / 3228.0 = 97.2% がカーネル実行時間である（§8.2 のストリーム統一と
  §8.3 の CUDA Graph の結果）。

## 8. tendency 以外のコスト（Amdahl 上限）

p=7/32³ と p=255/1 はどちらも owned volume point が 256³ = 16,777,216 個であり、
time-stepping ループの非 tendency カーネルは両者で同一である。

| kernel | 役割 | 理論バイト | nsys 実測 | 実効 BW | 対 7.9 TB/s | ncu DRAM% |
|---|---|---:|---:|---:|---:|---:|
| `main_64` | `q0 ← q`（SSP-RK の step 先頭保存） | 268.4 MB | 87.97 µs | 3.05 TB/s | 38.6% | 22.6 |
| `main_87` | RK 更新 `q = a*q0 + b*(q + dt*dqdt)` | 536.9 MB | 106.18 µs | **5.06 TB/s** | 64.0% | 42.1 |
| `main_99` | min/max reduction（`output_interval` ごと） | 134.2 MB | 116.0 µs | 1.16 TB/s | 14.6% | 41.4 |
| `update_halo_233` | halo 更新（RK stage ごと） | — | 4.7–6.1 µs | — | — | 21–29 |

1 step あたりの非 tendency GPU 時間は **約 422 µs**
（`q0` コピー 88 + RK 更新 3×106 + halo 3×5）。したがって:

| 条件 | tendency / step | 非 tendency / step | 非 tendency 比率 |
|---|---:|---:|---:|
| p=7 `CUDAFORTRAN_FUSED` | 1138 µs | 422 µs | **27.1%** |
| p=255 `CUDAFORTRAN_GEMM_FUSED` | 3464 µs | 430 µs | 11.0% |
| （参考）p=7 `OPENACC_ASIS` | 3026 µs | 422 µs | 12.2% |

**tendency を 2.66× 速くした結果、p=7 では GPU 時間の 4 分の 1 以上が
tendency 以外になった。** RK 更新は既に 5.06 TB/s（HBM3e 実効としてはかなり良い）
なので削り代は小さい。`q0 ← q` は 3.05 TB/s と低く、alignment / vector load-store /
他カーネルとの融合の余地がある。ただし **SSP-RK の意味と `q0` の寿命は変更不可**。

### 8.1 追記: `q0 ← q` を stage 1 の RK 更新に融合した（2026-08-25）

上の「他カーネルとの融合の余地」を実施した。SSP-RK の stage 1 では定義上
`q0 == q` なので、独立していた `q0 ← q` カーネルを stage 1 の更新カーネルに
畳み込める。`q0` の寿命も式の形も変えていない（`main.f90`）。

| | 融合前 | 融合後 |
|---|---:|---:|
| `main_64`（`q0 ← q`）| 268.4 MB / 88 µs | **削除** |
| stage 1 の RK 更新 | 536.9 MB | 671.1 MB（`q0` ストア +134.2 MB）|
| 非 tendency / step | 約 422 µs | **約 320 µs**（実測差 92-103 µs）|

実測の Main 時間は p=7 TC で 1.4146 → 1.3115 s、p=7 CUDA core で
1.5473 → 1.4520 s、p=255 `GEMM_FUSED` で 4.1889 → 4.0968 s
（`nstep=1000`, 各 3 回平均、login node）。`Cal_tend` と `CUDA device` は不変。
全点ビット一致の検証を含む詳細は `execution_times.md` の追記 4 にある。

p=7 `CUDAFORTRAN_FUSED_TC` での非 tendency 比率は 33% → **27%** に下がった
（tendency 851 µs / step に対し約 320 µs）。残るのは RK 更新 3 回分で、
これは既に 5.06 TB/s に達しているため、次の削り代は
min/max reduction（1.16 TB/s、`output_interval` ごと）と halo 更新である。

### 8.2 追記: カーネル間の GPU アイドルを消した（2026-08-25）

§8.1 の後も、非 tendency の wall 時間（`Main - Cal_tend` = 422 µs/step）が
非 tendency カーネルの device 時間（約 336 µs/step）より 86 µs/step 大きかった。
nsys（Slurm job `44070`, `nstep=60`, p=7 `FUSED_TC`）で 1 stage を追うと、
差はカーネルの**間**にあった。

```
tendency 279 µs │ 18 µs │ RK 更新 107 µs │ 14 µs │ halo 5 µs │ 14 µs │ 次の tendency
                 ↑ 空き            ↑ 空き           ↑ 空き        = 46 µs/stage = 139 µs/step
```

host は 1 stage に 4 回ブロックしていた。`!$acc parallel loop` に `async` が
無いので nvfortran が launch ごとに `cuStreamSynchronize` を出し、さらに
tendency ラッパが device 時間を読むために毎 stage `cudaEventSynchronize` していた。
どちらも次の launch を前のカーネル完了後まで遅らせるので、launch レイテンシが
毎回むき出しになる。

対策は 3 つで、いずれも演算と launch 順序を変えない。

1. 時間発展ループの OpenACC 領域を `async(ACC_QUEUE)` にし、host が値を読む
   直前だけ `wait`。
2. CUDA Fortran / C++ / cuBLAS / CUTLASS の全カーネルを、その OpenACC キューの
   ストリーム（`acc_get_cuda_stream`）に載せる。同一ストリームなので順序は不変。
3. tendency の CUDA event を同じ呼び出しで読まず、1 回後ろの呼び出しで読む
   （イベント 2 面持ち）。

| | 変更前 | 変更後 |
|---|---:|---:|
| stage あたりの gap 合計 | 46.3 µs | 16.8 µs |
| step あたりの gap | 138.9 µs | 50.4 µs |
| `cuStreamSynchronize` 回数（`nstep=60`）| 733 | 9 |
| Main: p=7 `FUSED_TC` | 1.3099 s | **1.2072 s** |
| Main: p=7 `FUSED` | 1.4412 s | **1.3444 s** |
| Main: p=255 `GEMM_FUSED` | 4.0890 s | **3.9601 s** |

device 時間は不変（p=7 TC 0.8523 → 0.8518 s）。残る 50 µs/step は 1 step
9 回の launch turnaround で、減らすには CUDA Graph 化のように launch 自体を
減らすしかない。詳細・検証・失敗した最初の方針は `execution_times.md` 追記 5。

**この変更以降、`Cal_tend` と `Volume derivate + surface lift` は tendency の
wall 時間ではない**（host が同期しなくなったため、キュー済みの device 仕事の
待ち時間を含む）。カーネル単体は `CUDA device *`（CUDA event）を見ること。

### 8.3 追記: 1 step を CUDA Graph にした（2026-08-25）

§8.2 で残った 50 µs/step は 1 step 9 回の launch turnaround そのものなので、
§12 の項目 5 に挙げていた **CUDA Graph 化**を実装した。SSP-RK3 の 1 step
（halo 更新・tendency・RK 更新 × 3 stage）を `cudaStreamBeginCapture` /
`cudaStreamEndCapture` で 1 回捕捉し、以降は `cudaGraphLaunch` で再生する。
namelist の `UseCudaGraph = .true.` で有効。カーネル・引数・順序・データは
一切変えていない。

捕捉は 2 step 目で行う。capture 中は何も実行されないので、`dqdt` を host が
読む 1 step 目は直接 launch する必要があるためである。再生は Fortran の
ラッパを通らないので、**このモードでは tendency の CUDA event 時間を採れない**
（`not measured (graph)` と表示される)。捕捉対象は ACC_QUEUE のストリームに
すべてのカーネルが乗っている経路に限られるので、tendency が default queue の
OpenACC 領域を使う `OPENACC_ASIS` / `OPENACC_SPLIT` / `CUDAFORTRAN_SPLIT` では
警告を出して無効化する。

| 入力 / パス | graph なし | graph あり | 差 | µs/step |
|---|---:|---:|---:|---:|
| p=7 `Ne=32^3` `CUDAFORTRAN_FUSED_TC` | 1.2038 | **1.1716** | −2.7% | −32 |
| p=7 `Ne=32^3` `CUDAFORTRAN_FUSED` | 1.3441 | **1.3104** | −2.5% | −34 |
| p=255 `Ne=1` `CUDAFORTRAN_GEMM_FUSED` | 3.9730 | **3.8545** | −3.0% | −119 |
| p=255 `Ne=1` `CUDAFORTRAN_GEMM_CUTE` | 4.2646 | **4.1328** | −3.1% | −132 |
| p=255 `Ne=1` `CUDAFORTRAN_FUSED` | 15.346 | **15.288** | −0.4% | −59 |

削れた絶対量はどのパスでもほぼ同じで、相対利得の差はカーネル時間の差である。
詳細・数値検証・イベント計測コストの分離は `execution_times.md` 追記 7。

---

### 8.4 追記: p=255 の separable lift を 1 本のカーネルにした（2026-08-25）

§5.1 の内訳で lift は 176 µs / tendency call、§6 の ncu では
`d884gemm_*`（lift）の SM throughput が 27.8–37.1% しかない。これは
`K=2` の rank-2 更新を d884 TensorOp mainloop に載せていたためで、実体は
`lift_out`（`Ne=1`, `Np=256³` で 134 MB）への往復である。3 本の GEMM は
`beta=1` で累積するので write 134 + rw 268 + rw 268 = **670 MB**、
176 µs に対して約 3.8 TB/s、つまり帯域律速だった。

面データは 6 × `nq2` × 8 B = 3 MB しかないので、

```
lift(i,j,k) = Lift1D(i,2)*fb2(j,k) + Lift1D(i,4)*fb4(j,k)
            + Lift1D(j,1)*fb1(i,k) + Lift1D(j,3)*fb3(i,k)
            + Lift1D(k,5)*fb5(i,j) + Lift1D(k,6)*fb6(i,j)
```

を 1 スレッドでまとめて評価し `lift_out` を 1 回だけ書く
`separable_lift_kernel` に置き換えた。GEMM 3 本と pack/copy 3 本が消え、
lift 側の DRAM は 670 → 134 MB になる。加算順は GEMM と同じ `(x+y)+z` に
そろえたので、**旧実装とビット一致**する（`execution_times.md` 追記 8）。

| `DqdtKernel_Type` | 変更前（`514853f`） | 変更後 | |
|---|---:|---:|---:|
| `CUDAFORTRAN_GEMM_FUSED` | 3.971 | **3.635** | −8.5% |
| `CUDAFORTRAN_GEMM_CUTE` | 4.279 | 3.954 | −7.6% |
| `CUDAFORTRAN_GEMM` | 4.241 | 3.962 | −6.6% |

`nstep=1000`、`UseCudaGraph = .false.`、login node。
**§5.1 と §6 の lift 行は変更前の値である。**

残っているのは z-epilogue が `lift_out` を読む 134 MB で、これは lift を
epilogue 内で面から直接評価すれば消える（§12-6）。

---

### 8.5 追記: p=255 の lift を z-epilogue に畳み込んだ（2026-08-25）

§8.4 の後も z-epilogue は `lift_out`（134 MB）を読んでいた。z GEMM の
ユーザ問題は `(m=Nq², n=Nq)` の column-major で、CUTLASS の
`GemmBatched` は ColumnMajor C を A/B 入れ替えの row-major 問題として解く。
したがって **epilogue タイルの row は z 添字 `k`、column は xy 面添字
`p = i + j*Nq`** であり、6 枚の面（計 3 MB）から lift をその場で
組み立てられる。`lift_out` は生成も読み出しも無くなった。

ここで測定が教えたことが 1 つある。**素直に書いた版は −0.6% にしかならない。**
出力 1 点ごとに `p % Nq` と `p / Nq` を計算していたためで、`Nq` は実行時値
なので整数除算になる。z GEMM は SM throughput 79.8% の演算律速だから、
epilogue に足した整数演算がそのまま効く。
`PredicatedTileIterator::operator++` は `thread_start_row_` しか進めない
ので、column に依存する量はすべて `kIterations` ループを通じて不変である。
それをループ外に括り出すと −4.7% になった。

| 版 | Main | `CUDA device GEMM fused` |
|---|---:|---:|
| `514853f`（3 本の lift GEMM） | 3.971 | — |
| §8.4（`separable_lift_kernel`） | 3.635 | 3.266 |
| lift を epilogue へ（除算そのまま） | 3.615 | 3.244 |
| **lift を epilogue へ（column 不変量を hoist）** | **3.463** | **3.081** |

`nstep=1000`、graph off、login node。3000 tendency call なので device は
1089 → 1027 µs/call。`514853f` からの通算 Main **−12.8%**。
旧実装と**ビット一致**（`execution_times.md` 追記 9）。

**§5.1 と §6 の lift 行・z GEMM 行はいずれも変更前の値である。**
z GEMM のレジスタ 242 本と SM throughput 79.8% は epilogue を厚くした分
再測定が要る（§12-1）。

これで §12-6 の「p=255 の lift」は完了である。GEMM 系に残る帯域律速の
独立カーネルは `volume_flux_kernel`（150 µs、DRAM 66%）のみ。
（→ §8.6 でこれも 125.9 µs / DRAM 83% になった。）

---

### 8.6 追記: `volume_flux_kernel` のロードをストアより前にまとめた（2026-08-25）

§8.5 の時点で GEMM 系に残る帯域律速の独立カーネルは `volume_flux_kernel`
だけだった。「消せる DRAM トラフィックが融合の得の上限」（§6 末尾）という
見立てからは、このカーネルはトラフィック最小なので手の付けようがない。
実際 ncu で採ると `dram__bytes_read.sum` は理論値の **1.000 倍**、ld / st の
セクタ効率はどちらも 100% で、無駄なトラフィックは 1 バイトも無かった
（job 46163、`p=255 Ne=1`、`CUDAFORTRAN_GEMM_FUSED`、commit `d7b1853`）。

同じ測定が別のことを示していた。**どのユニットも飽和していない**
（DRAM 65.7%、L1 52.5%、L2 36.8%、SM 33.7%）。つまり帯域律速ではなく
レイテンシ律速である。加えて `smsp__inst_executed_op_global_ld.sum` が
1 warp あたり **6 命令**で、あるべき 4 命令より多かった。

```fortran
flux_x(idx) = q(idx)*u(idx)
flux_y(idx) = q(idx)*v(idx)
flux_z(idx) = q(idx)*w(idx)
```

この書き方だと nvfortran はロードとストアを交互に発行し、`q` を 3 回
読み直す。余分な 2 回は L1/L2 に当たるので DRAM read は理論値のままだが、
レイテンシ律速のカーネルでは発行スロットがそのまま時間になる。
4 本のロードをストアより前にまとめて発行させると（job 46183）:

| | 変更前 | 変更後 |
|---|---:|---:|
| global ld 命令 / warp | 6 | **4** |
| ncu duration | 173.2 µs | **135.0 µs** |
| DRAM throughput | 65.7% | **83.4%** |
| L1 / SM throughput | 52.5 / 33.7% | 67.0 / 43.2% |
| register / occupancy | 20 / 84.1% | 24 / 79.9% |
| **nsys duration** | **150.6 µs** | **125.9 µs**（−16.4%）|

893 MB / 125.9 µs = **7.09 TB/s**、参照ピーク 7.9 TB/s の **90%**。
これはこのリポジトリのどのカーネルより高い実効帯域である
（§8 の RK 更新 5.06 TB/s、`q0 ← q` 3.05 TB/s と比較のこと）。

Main は p=255 `GEMM_FUSED` 3.4469 → **3.3702** 秒（−2.2%）、
`GEMM` 3.9403 → 3.8713、`GEMM_CUTE` 3.9539 → 3.8820、
p=7 `CUDAFORTRAN_SPLIT` 2.7172 → **2.6440** 秒（−2.7%）。
`CUDAFORTRAN_FUSED` / `FUSED_TC` はこのカーネルを使わないので変化しない。
旧実装と**ビット一致**（`execution_times.md` 追記 10）。

**`q` だけをレジスタに退避した版はまったく効かない**（3.080 秒のまま）。
効いているのは共通部分式の除去ではなく、ストアを挟まないロード窓のほうで
ある。1 スレッド 2 / 4 / 8 点に増やして MLP を稼ぐ版も試したが、2 点は同値、
4 / 8 点はわずかに悪化した。

同じジョブで x/y/z GEMM も `d7b1853` で採り直した（§6 の z GEMM 行は lift を
epilogue へ移す前の値なので、以下が現行値である）:

| kernel | nsys | SM% | DRAM% | L1% | reg | occ% |
|---|---:|---:|---:|---:|---:|---:|
| x GEMM `Gemm<64,128,16>` | 249.7 µs | 87.9 | 10.1 | 34.3 | 212 | 12.2 |
| y GEMM `GemmBatched<64,64,16>` | 243.5 µs | 89.2 | 6.3 | 47.9 | 130 | 18.5 |
| z GEMM + assembly + lift | 339.1 µs | 72.5 | 20.1 | 56.5 | 254 | 12.2 |

z GEMM の SM throughput は lift を載せたことで 79.8 → 72.5%、レジスタは
242 → 254 本になった（§12-1 の再測定はこれで済んだ）。x/y GEMM が 493 µs の
あいだ DRAM を 6–10% しか使っていないことは、§12 に挙げた
「帯域律速カーネルを 2 本目のストリームで GEMM の裏に隠す」案の前提になる。

### 8.7 追記: 境界流束を 2 本目のストリームで GEMM の裏に隠した（2026-08-25）

§12-7 に挙げた案を実測した。x GEMM（249.7 µs、SM 87.9%、DRAM 10.1%）と
y GEMM（243.5 µs、SM 89.2%、DRAM 6.3%）の 493 µs のあいだ DRAM は空いており、
方向ごとの依存も独立している。そこへ帯域律速のカーネルを流し込む。

**採用できたのは `elembnd_flux_kernel`（19.6 µs、105 MB）だけである。**
volume flux を方向で割って side stream に載せる版（`flux_y,z` を隠す、
`flux_z` を隠す）はどちらも効かなかった。理由はレジスタで、x GEMM は
212 reg × 128 thread の CTA を 2 個/SM 走らせて 54,272 / 65,536 本を占め、
**SM あたり 11,264 本しか残さない**。24 reg の flux ブロック（256 スレッド =
6,144 本）は 1 個しか同居できず、side 側の並列度は単独時の約 1/8 になる。
`flux_y,z` の 671 MB を 250 µs の窓で流すには 2.7 TB/s 必要だが、その並列度では
出ない。隠れないうえに DRAM を奪って GEMM を遅くする。

C 案の重なりは nsys で直接確認した（job 46362）。

```
strm=14  start=3436141.0us  dur=253.0us  end=3436394.0us  x GEMM
strm=15  start=3436359.3us  dur= 26.9us  end=3436386.2us  elembnd_flux_kernel
```

elembnd は x GEMM の区間に完全に収まる。代償は elembnd 自身が 19.6 → 26.9 µs、
x GEMM が 249.7 → 253.0 µs で、差し引き **−15 µs/call**。開始が x GEMM の
218 µs 後であることから、ブロックは GEMM の最後の wave の隙間に入っている。

device 時間（`CUDA device GEMM fused`、3000 call）は 3.0058 → 2.9646 秒、
1001.9 → 988.2 µs/call。Main は graph off で 3.3723 → **3.3293** 秒（−1.3%）、
`GEMM_CUTE` 3.8820 → 3.8368、`GEMM` 3.8713 → 3.8613。旧実装と**ビット一致**。

**CUDA Graph の replay では逆に +5 µs/call の損になる**ので、`UseCudaGraph` の
ときは 1 本に戻す（`cuda_dg_set_side_stream(.false.)`、`main.f90`）。1 本に
戻すときは elembnd を volume flux の**前**に置き直すことも必要で、これを怠ると
graph on が +0.6% になる。elembnd は `VMapM`/`VMapP` で `q` を gather するので、
volume flux が `q,u,v,w` を L2 に流す前のほうが安い。

p=7 の `CUDAFORTRAN_GEMM` でも無効にしている。そこでは elembnd が 181 µs で
volume GEMM は 8×8 の 24 launch なので、隠す先の窓より隠すものが大きく、
重ねると +3.1% になる。

これで §12-7 は完了である。**ここで得た一般則: SM 律速の GEMM の裏に隠せる
帯域律速カーネルの大きさは、GEMM が SM あたりに残すレジスタ本数で決まる。**
DRAM に空きがあることは必要条件でしかない。

### 8.8 追記: 全カーネルのロード監査と SSP-RK 更新の 1 次元化（2026-08-25）

§8.6 で `volume_flux_kernel` が 1 warp あたり 6 命令の global load を出していた
ことが分かったので、同じ欠陥が他に無いかを**全カーネルで測った**。指標は
`smsp__inst_executed_op_global_ld.sum ÷ warp 数` と、ソース上の相異なる
ロード数の突き合わせである（job 46402 / 46417）。

**CUDA Fortran / CUDA C++ のカーネルはすべて最小だった。**
`elembnd_flux_kernel` 14.00（最小 14）、`separable_lift_kernel` 12.00（12）、
`dqdt_assembly_kernel` 7.00（7）、`surface_lift_p7_kernel` 12.00（12）、
`tendency_fused_p7_kernel` 35.50（35.5）、`tendency_fused_p7_tc_kernel` 32.50
（32.5）、`update_halo` 2.00（2）。`volume_deriv_p7_kernel` だけ 5.00 対 3.25 だが、
差は述語化された `D1D`/`D1D_tr` が全 warp で発行されるためで、sector は 26/warp
＝理論どおり、メモリトラフィックは無い。

なお `tendency_fused_p7_kernel` は `q(idx1)*u(idx1)`, `q(idx1)*v(idx1)`,
`q(idx1)*w(idx1)` と `volume_flux_kernel` と同じ書き方をしているのに最小である。
違いは間に挟まるストアが **shared** であることで、§8.6 の再ロードは
global ストアが assumed-size dummy の別配列に当たるかもしれない、という
エイリアス解析の保守性から来ていたと分かる。

**外れたのは時間発展ループ側の OpenACC カーネル 2 本だけだった**
（SSP-RK 更新、stage 1 が 4.00 対 2、stage≥2 が 5.00 対 3）。余分な 2 本は
`rk_a(stage)` / `rk_b(stage)` で、`stage` が実行時値なので配列参照が
1 スレッド 1 ロードになる。ところが**それをスカラーに読み出しても
−0.2〜0.5% にしかならない**。2 本とも全スレッドが同じアドレスを読むので
L1 に当たり、費やしているのは発行スロットだけだからである。

**本当の律速は `collapse(2)` だった。** DRAM 41.7% に対し SM throughput が
62.7% と高く、3 ロード 1 ストア 3 演算のカーネルとしては説明がつかない。
`collapse(2)` は平坦化したスレッド番号から 2 つの添字を復元するのに
**実行時値 `Np` による整数除算**を要求する（§8.5 で z-epilogue に対して得た
のと同じ罠）。owned 領域 `q(:,1:Ne)` は連続なので 1 次元ループに書き直せば
除算は消える。

| | `collapse(2)` | 1 次元 |
|---|---:|---:|
| ld/warp（stage 1 / stage≥2） | 4.00 / 5.00 | **2.00 / 3.00** |
| ncu duration | 152.5 / 154.1 µs | **91.2 / 86.7 µs** |
| SM throughput | 62.7% | **35.3%** |
| DRAM throughput | 41.7% | **74.2%** |
| DRAM バイト | 479.7 / 509.4 MB | 479.2 / 509.8 MB（不変）|

演算律速から帯域律速に変わり、カーネル時間は約 321 → 185 µs/step になった。

（2026-08-26 追記: 上の DRAM throughput は **ncu の値をクロック換算せずに
読んだもの**である。ncu は SM クロックを 1.08 GHz に固定するので、実運用
クロックでは 79–87% になる。結論は変わらないが、「まだ 74% なので伸びしろが
ある」とは読まないこと。§13.1）
**tendency に触っていないので全パスが同じだけ得をする。** Main は p=7
`FUSED_TC` 1.208 → **1.131** 秒（graph on では 1.171 → **1.083**）、
p=7 `FUSED` 1.345 → 1.250、p=255 `GEMM_FUSED` 3.329 → **3.232**（graph on
3.289 → **3.192**）、`OPENACC_ASIS` でも 3.50 → 3.41。旧実装と**ビット一致**
（`execution_times.md` 追記 12）。

これで §8 の「非 tendency は削り代が小さい」という見立ては更新された。
`main_87` を 5.06 TB/s と評価して「HBM3e 実効としてはかなり良い」と書いたが、
**それは実効帯域ではなく整数除算で頭打ちになっていた値**である。

### 8.9 追記: p=255 z GEMM の shared store バンクコンフリクトを消した（2026-08-26）

`tma_survey.md` §2.2 が残していた標的。p=255 最速パスの単独最大カーネル
（z GEMM、tendency の 31.6%）で shared store wavefronts の 52% が
バンクコンフリクトだった。測定は Slurm job `49543` / `49546`、
p=255 `Ne=1`、`-s 6 -c 3`、`UseCudaGraph = .false.`、凍結実行ファイル
`scale-dg_extraction.zbank0`（`9eff7f8`）と `.zbank1`（修正後）。

**帰属が違っていた。** §2.2 は CUTLASS 標準の `MmaMultistage` が A/B タイルを
書くところと推定していたが、sm80 multistage の global→smem は `cp.async`
（LDGSTS）で STS を 1 命令も出さない。実体は **epilogue のアキュムレータ
smem ステージング**で、命令数がぴたりと合う: TB 64×32 / warp 32×32 /
inst 8×8×4 の 2 warp なので `TensorOpPolicy` の `kIterations = 4`、
`TileIteratorTensorOp::store` は毎回 `OperatorCount::kColumn = 4` 本の
STS.128 を出し、1 block = 32 STS、8192 block で **262,144** = 実測 requests。

原因は `DefaultEpilogueTensorOp` の
`Padding = MatrixShape<0, 64/sizeof_bits<ElementAccumulator> * 4>`
（`double` では 4）。ステージングタイルは 16 行 × 36 doubles、行ストライド
288 B = 72 word で `72 mod 32 = 8` となり、連続 2 行の 16 バンク幅が 8 バンク
重なって **2-way**（理想 4 wavefront/命令が 8.47 になる）。Padding を **8** に
すると行ストライド 40 doubles = 80 word、`80 mod 32 = 16` で 2 行が 32 バンクを
ちょうど 1 回ずつ覆う。

実装は `cutlass_z_gemm_assembly.h` の `RepadEpilogue`（既定 Epilogue の公開
typedef から `Padding` だけ差し替えるエイリアス）を `cuda_cutlass_gemm_fused.cu`
の z GEMM に当てただけ。epilogue の smem は mainloop の 49,152 B と union
なので確保量も occupancy も不変。出力は旧実装と**ビット一致**で、
点ごとに変化する係数（`SCALE_DG_VARYING_COEFF=1`）での `CUDAFORTRAN_GEMM`
との比較も `Ne=1` / `Ne=2` の両方で相対 4.2e-16。

| z GEMM | 前 | 後 |
|---|---:|---:|
| shared store wavefronts | 2,214,315 | **1,184,955**（理想 1,048,576）|
| shared store バンクコンフリクト | 1,165,739 | **136,379**（−88%）|
| shared load バンクコンフリクト | 211,956 | 212,406 |
| LDGSTS バンクコンフリクト | 3,869,027 | 3,824,290 |
| L1/TEX throughput | 56.33% | 55.13% |
| Compute (SM) throughput | 72.30% | 72.18% |
| duration（ncu クロック） | 597.9 µs | 595.4 µs |

**コンフリクトは消えたが時間は動かない。** login node、
`bench_runs/p255_gemm_fused.conf`（`nstep=1000`）で版を交互に 3 ラウンド:

| 版 | Main | `CUDA device GEMM fused` |
|---|---:|---:|
| 前（`9eff7f8`） | 3.2315 | 2.9647 |
| 後 | 3.2313 | 2.9642 |

差はラウンド間ばらつきの内側である。`tma_survey.md` §2 の stall 内訳
（wait 36.6% / math pipe 20.4% / long scoreboard 11.3%）どおり、このカーネルは
shared 経路では律速されていない。**ncu の「推定改善余地 30.5%」は shared 経路
単独の上限であって、カーネル時間の予測ではない**——これは §10 の一般則に足すべき
教訓である。それでもコストがゼロでビット一致なので、変更自体は残した。

同じ job で x / y GEMM も測った。3 本とも epilogue 側（STS）と mainloop 側
（LDGSTS）の両方にコンフリクトを持つ:

| カーネル | duration | SM tput | STS コンフリクト | LDGSTS コンフリクト |
|---|---:|---:|---:|---:|
| x GEMM `Gemm` 64×128 | 462.0 µs | 88.15% | 1,092,914 | 632,474 |
| y GEMM `GemmBatched` 64×64 | 472.3 µs | 88.20% | 1,122,049 | 2,092,773 |
| z GEMM `DqdtAssembly` 64×32（前） | 597.9 µs | 72.30% | 1,165,739 | 3,869,027 |

x / y は device 側 API を通しているので `Padding` を差し替えられず、どちらも
SM 88% の演算律速なので手を付けていない。LDGSTS 側は `MmaPolicy` の smem
padding が `MatrixShape<0,0>` 固定で、64 bit multiplicand レイアウトの swizzle も
非テンプレートのハードコードなので **CUTLASS 側に調整の余地が無い**。
z の結果から見て、どちらも時間では報われない公算が大きい。

## 9. 試して不採用にした最適化と、その理由

| 試行 | 結果 | 判断 |
|---|---|---|
| `q*vel` を GEMM mainloop に融合（`MulPairIterator`） | p=255 で 3.91 s → **6.48 s**（1.66× 悪化） | **不採用**。消せる `volume_flux` は 0.45–0.5 s なのに、CTA ごとの `q`/`vel` 再読と dual `ld.global` で標準 global→shared パイプラインを壊し 2.5 s 以上を失う |
| volume flux を方向で割って 2 本目のストリームへ（`flux_y,z` / `flux_z` を x/y GEMM の裏に） | device 3.0058 → 3.0245 / 3.0034 秒 | **不採用**（§8.7）。x GEMM は SM あたり 11,264 本しかレジスタを残さず、24 reg の flux ブロックは 1 個/SM しか同居できない。671 MB を 250 µs で流すには 2.7 TB/s 要るがその並列度では出ず、隠れないうえに GEMM を遅くする |
| z-GEMM epilogue に assembly を融合 | 3.914 s → **3.603 s**（−8%） | **採用**。mainloop に触らず、DRAM 律速の独立カーネルだけを消す |
| epilogue の barrier をまとめる | maxabs ≈ 500（数値不正）、改善なし | 不採用。CUTLASS 標準 epilogue は smem スロットを iteration ごとに再利用しており、tile 対応が壊れる |
| auxiliary fragment の寿命短縮 | maxabs 2e-15（可）、3.60 → **3.70 s** | 不採用。load 直列化のレイテンシがレジスタ圧緩和を上回る |
| epilogue loop のフルアンロール | maxabs 3.6e-15（可）、3.60 → **3.69 s** | 不採用。register / code-size で不利 |
| 代表スカラー特殊化（`Escale(1,1,1)*u(1,1)` 等） | 高速だが**数値契約違反** | 撤回（`03551c7`）。速度に関係なく即 revert |
| cuBLAS FP64 emulation（Ozaki fixed-point） | 後の調査で API 判定と表示の誤りを修正。EAGER は p=7 で native の約131倍遅い | `nstep=1--10` でのみ比較。`cublas_emulation_survey.md` 参照 |
| SSP-RK 更新の `rk_a(stage)` / `rk_b(stage)` だけをスカラー化 | −0.2〜0.5% | 単独では**ほぼ無意味**（§8.8）。全スレッドが同一アドレスを読むので L1 に当たり、発行スロットしか消費していない。同じ関数の 1 次元化と併せて採用 |
| p=7 の grid-stride loop 除去（1 thread / 1 point） | 全体の律速は解消せず | 不採用。launch 構造と中間配列トラフィックが本体 |
| p=7 TC の face gather 前倒し（`VMapM`/`VMapP` の先行ロード） | 版 A（index だけ前倒し）0.850 → **0.868 s**（+2.1%）、版 B（セクションごと入れ替え）0.849 s（±0） | **不採用**（2026-08-25）。カーネルはレジスタ 32 本ちょうどで余裕がゼロなので、ptxas が先行ロードを元の位置へ押し戻す。詳細は `tc_paper_survey_2407.09621.md` §12 |

現在 z-epilogue に残しているのは、iteration あたり 2 sync、6 operand のまとめ読み、
`#pragma unroll(1)` の構成。

---

## 10. 一般則（本作業から得られた再利用可能な知見）

1. **入力の現在値と API の意味を混同しない。** 定数が入っていても配列は配列として扱う。
   定数入力ベンチだけでは scalarization バグを検出できない。
2. **高速化の前に数値契約を回帰テストで固定する。** 点ごとに係数を変えた比較
   （`SCALE_DG_VARYING_COEFF=1`）を全パスで自動化する。
3. **同じ DOF 数は同じ GPU 問題を意味しない。** 多項式次数と要素数で
   並列構造・行列次数・launch 数・再利用量が変わり、最適戦略が逆転する。
4. **ロード命令数を数える。** `smsp__inst_executed_op_global_ld.sum ÷ warp 数`
   をソース上の相異なるロード数と比べるだけで、コンパイラが再ロードしている
   カーネルが 1 回の ncu で分かる（§8.8）。ただし**差分が時間になるとは
   限らない**: 全スレッド同一アドレスのロードは L1 に当たるので発行スロット
   しか食わない。原因を突き止める手掛かりであって、それ自体が答えではない。
5. **実行時値による整数除算を疑う。** `collapse(2)` も epilogue の
   `p % Nq` も、実行時値で割る点は同じで、帯域律速に見えるカーネルを
   演算律速にしていた（§8.5 / §8.8）。連続領域を 1 次元ループで回れば消える。
6. **「DRAM %」だけを見て帯域律速と判定しない。** どのユニットも飽和していない
   （最大でも 66%）カーネルはレイテンシ律速であり、トラフィックが理論最小でも
   まだ速くなる。`volume_flux_kernel` は 1 行の書き方（ロードをストアより前に
   まとめる）で −16.4%、DRAM 83.4% になった（§8.6）。
7. **カーネル数削減は常に正義ではない。** 融合の得は「消せる DRAM トラフィック」で
   上限が決まる。GEMM mainloop を壊す融合はそれを大きく超えて損をする。
8. **中間配列は必ずしも無駄ではない。** p=255 の volume flux 配列は
   高効率 dense GEMM のための materialized input / cacheable preprocessing である。
9. **occupancy 単独で判断しない。** register、shared bank conflict、L1 throughput、
   instruction mix を同時に見る（p=7 TC 版は occupancy 97% で 1.28× 遅い）。
10. **理論 FLOP/byte と NCU 実測を分けて示す。** tile 再計算・cache hit・FMA 化で
   両者は一致しない。
11. **device-event と wall time を混ぜない。** 表に「launch/sync を含むか」を明記する。
12. **profiling は同一 input・同一 commit で行う。** 過去の scalar 特殊化版の
   プロファイルを現行版に適用しない。
13. **速くなったものだけ残す。数値不一致は速度に関係なく即 revert。**
14. **ncu の「推定改善余地 N%」はその経路単独の上限であって、カーネル時間の
   予測ではない。** z GEMM の shared store コンフリクトは推定 30.5% だったが、
   実際に潰すとコンフリクト −88% / store wavefront −47% に対して時間は
   ±0 だった（§8.9）。先に stall 内訳を見て、その経路で本当に止まっているかを
   確かめること。
15. **コンフリクトの帰属をカーネル名や直感で決めず、命令数で突き合わせる。**
   §8.9 の標的は「CUTLASS の mainloop」と推定されていたが、mainloop は
   `cp.async` なので STS を 1 命令も出さない。1 block あたりの STS 本数を
   テンプレートから数えると requests と完全に一致し、epilogue 側だと確定した。

---

## 11. 測定条件の記録

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GB200（RIKYU）1 基、CC 10.0、152 SM |
| 参考ピーク | FP64 40.1 TFLOP/s（CUDA-core / Tensor Core とも同値、§7 参照）、HBM3e 7.9 TB/s（RIKYU system document） |
| SM clock | 2062 MHz（`nvidia-smi --query-gpu=clocks.max.sm`） |
| 実行体 | `scale-dg_extraction`、commit `299a868`（`feature/cuda`） |
| ビルド | `module load nvhpc-hpcx` → `make clean; make CUDA=1 GPUFLAGS=-gpu=cc100`（nvcc `-arch=sm_100`） |
| 時間比較（表 4） | `nstep=1000`, `DGOptrKernel_OptType=OPT1`, Slurm job 41348, host c162 |
| nsys / ncu（表 5–8） | `nstep=10`, `output_interval=10`, `dt=1e-7`, Slurm job 43219, `job_all.sh` |
| ncu 追加採取（GEMM_FUSED p=255） | 同条件、Slurm job 43241 / 43254 / 43259 |
| §7 / §7.1 の採り直し（2026-08-26） | `63a4234` + §13.2、`nstep=20`、`conf_perf_p7.conf` / `conf_perf_p255.conf`、nsys job 49692（最速パス）と 49700（全パス）、DRAM バイトは ncu job 49699 |
| §13 の棚卸し（2026-08-26） | 同ツリー、`ncu --set full` job 49566、nsys job 49567、source ページ job 49589 |
| run + nsys + ncu 追加採取（GEMM_CUTE p=255） | 同条件、Slurm job 43244 / 43246 / 43254 |
| nsys コマンド | `nsys profile --trace=cuda,nvtx,openacc --sample=none --resolve-symbols=false --stats=true` |
| ncu コマンド | `ncu --set basic -k regex:<kernel> -s <instances-1> -c 1`（`gen_ncu_cmds.sh` が nsys の `cuda_gpu_kern_sum` から自動生成） |
| p=7 ケース | `NeX=NeY=NeZ=32`, `PolyOrder=7`（Nq=8, Np=512, 体積 DOF 256³） |
| p=255 ケース | `NeX=NeY=NeZ=1`, `PolyOrder=255`（Nq=256, Np=256³, 体積 DOF 256³） |

補足:

- **物理領域は常に 1×1×1。** `NeX/NeY/NeZ` を増やしても領域は広がらず要素が細かくなる。
  通常の weak scaling ではない。
- `nstep=10` のアプリタイマは cuBLAS/CUTLASS 初期化を含むため、p=255 GEMM 系で
  nsys カーネル総和（1.25 ms/call）より大きい値（1.90 ms/call）を示す。
  `nstep=1000` では 1.29 ms/call に収束する。**定常値は nsys 側を使うこと。**
- p=7 FUSED は両者が一致する（app 384.8 µs、nsys 379.5 µs、nstep=1000 で 383.4 µs）。

### 11.1 測定バッチの補完と注意点

- `CUDAFORTRAN_GEMM_FUSED` p=255 は job 43219 の時点で ncu が採れていなかった。
  `gen_ncu_cmds.sh` が nsys の 100 文字で切り詰められた CUTLASS シンボルから
  一意なパターンを作れず `duplicate ncu match pattern 'MmaMultistage'` で停止した
  （`slurm-43219.out:1269`）。3 つの CUTLASS カーネルが可視部分では
  `MmaMultistage` しか共有トークンを持たなかったことが原因。
  **→ Slurm job 43241 で手書きパターンにより再取得済み**。
  全 15 カーネルを 1 launch ずつ採取し、各レポートの kernel 名と
  `GemmShape` テンプレート引数で取り違えが無いことを確認した。
  パターンは ncu の `-k regex:` に合わせて colon-free
  （`Kernel<cutlass..gemm..kernel..Gemm<` のように `::` を `..` で表記）としている。
- `CUDAFORTRAN_GEMM_FUSED` p=7 と `CUDAFORTRAN_SPLIT` p=255 は
  実装上サポート外のため `ERROR STOP`（想定どおり）。
- `CUDAFORTRAN_GEMM_CUTE` は job 43219 の `job_all.sh` の対象に含まれていなかった。
  **→ job 43244（run + nsys）と job 43246（ncu）で取得済み**。
  CUTE は volume GEMM の 2 本が同じ `GemmBatched` テンプレートで、
  nsys の可視名が完全に一致するため、この 2 本だけは
  `-s 58 -c 2` で 1 レポートに 2 launch まとめて採取し、
  各ブロックの `GemmShape<64,64,16>` / `<64,32,16>` で識別する。
  この扱いは後述のとおり `gen_ncu_cmds.sh` 本体に取り込んだ。
- 3 つの ncu セット（43219 / 43241 / 43246）は同一実行体・同一入力条件で採取しており、
  共通カーネル（`volume_flux`, `main_64/87/99`, `halo` など）の値が
  1–2% 以内で一致することを確認済み。
- **`gen_ncu_cmds.sh` に上記 2 種のフォールバックを実装済み**なので、
  `job_all.sh` を再実行しても GEMM_FUSED / GEMM_CUTE で停止しない。
  - 可視名が異なる場合: colon-free な sanitized prefix 正規表現を伸ばして一意化する
    （例 `void.cutlass..Kernel<cutlass..gemm..kernel..Gemm<`）。
  - 可視名が同一の場合: そのグループを 1 コマンド `-c <グループ数>` で採取し、
    1 レポート内に各カーネルのブロックを得る。
  既存 43219 の全ログで生成されるレポート名が既存ファイルと完全一致すること（無退行）、
  および新規 4 パターンが ncu で意図どおりのカーネルにマッチすることを
  job 43254 / 43259 で実測確認した。
- `output/` の GEMM_FUSED / GEMM_CUTE の ncu レポートは **`gen_ncu_cmds.sh` の
  命名に統一済み**（手動採取時の重複ファイルは削除）。
  CUTE の 2 本の volume GEMM は
  `CUDAFORTRAN_GEMM_CUTE_p255_ncu_GemmBatched.ncu-rep` の
  1 ファイルに 2 ブロックとして入っており、`GemmShape<64,64,16>`（y）と
  `<64,32,16>`（z）で識別する。§6 の値はこの統一後のファイルから再取得したもので、
  手動採取時との差は 0.3 ポイント以内（run-to-run のばらつき）である。

---

## 12. 次に価値のある調査

1. ~~`CUDAFORTRAN_GEMM_FUSED` / `CUDAFORTRAN_GEMM_CUTE` の ncu 採取~~
   → **完了（job 43241 / 43244 / 43246）**。
   次は z-epilogue のレジスタ 156 → 242 本の増加を抑えられるか。
   occupancy は CUTE と同じ 12.2% なので、効くとすれば SM throughput の
   −7.1 ポイント分である。過去に試した fragment 分割・フルアンロールは
   いずれも悪化しているため、`--set full` の Memory Workload Analysis /
   Scheduler Statistics / Warp State を取ってから判断する。
2. **p=7 の L1 帯域律速の解消。** occupancy 側は §8 で解決済みで、
   TC 版・CUDA core 版とも L1/TEX 95% 台が上限として残っている。
   M 側 gather を shared に移す案は `tc_paper_survey_2407.09621.md` §9 で
   実測し**不採用**（global sectors −26% に対し shared バンクコンフリクト 20 倍、
   正味 3.7% 遅い）。残る筋は、contraction のアクセスと 6 面の face gather を
   同時にコンフリクトフリーにする shared レイアウトが存在するかどうかで、
   これはカーネル改造ではなくレイアウト探索の問題である。
   → **§10 で決着**。レイアウトは存在し実機でもコンフリクトを消すが、
   それでも遅い。§10 はここで「次の標的は L1/TEX ではなく整数・アドレス演算」
   と書いたが、**その見立ては §11 の実測で外れた**（2026-08-25）。整数命令
   だけを 14% 削った版は end-to-end で無変化である。効いたのは global
   アクセスの**キャッシュライン数**（sector 数が同じでもライン数が 2 → 4 に
   なると 11.6% 遅い）で、x/y の導関数を shared を通さずレジスタに置き、
   mma の出力を転置して epilogue を coalesce させた版が採用になった
   （device 0.8518 → 0.8488 秒、ncu 単発では 434.6 → 405.5 µs）。
   その版で残る律速は **global load のレイテンシ**（`long scoreboard`
   25.5 → 35.0）であり、sector 数・L2・DRAM は不変である。
   詳細は `tc_paper_survey_2407.09621.md` §11。
   → **さらに §12 で決着（2026-08-25）**。§11.6 が挙げた「face gather の
   前倒し」も**外れ**である。`VMapM`/`VMapP` の先行ロードは 2.1% 遅く、
   セクションごとの入れ替えは ±0 だった。原因は 8 ブロック/SM を保つための
   レジスタ 32 本という枠で、ptxas に先行ロードを保持する余地が構造的に無い。
   加えて occupancy 96.9%・64 warp/SM では、そもそもスレッド内の
   memory-level parallelism を増やしても意味がない。`long scoreboard` の
   増加は「発行できるロードが足りない」ではなく **L1 で待たされている**
   という意味であり、残る余地は **L1 トランザクション数の削減**側にしかない。
   その方向の案（§9 の M 側 gather shared 化、§10 のレイアウト探索）は
   すでに実測で不採用になっている。**p=7 の tendency カーネルは打ち止めに
   近い。**
   あわせて、ncu 単発 launch と実運用 launch では最適化の優劣が逆転しうる
   ことが分かった（同 §10.5、§11.5）。ncu で絞った候補は
   必ず `nstep=1000` の end-to-end で確認すること。
3. ~~**`q0 ← q` の改善。**~~ → **完了（2026-08-25、§8.1）**。stage 1 の RK
   更新に融合してカーネルごと削除した。非 tendency は約 422 → 320 µs/step。
   次に残るのは min/max reduction（1.16 TB/s）と halo 更新だが、
   前者は `output_interval` ごとなので現行入力での寄与は小さい。
4. ~~**p=7 と p=255 の間の crossover point 測定。**~~ → **成立しないので取り下げ
   （2026-08-25）**。CUDA Fortran の専用カーネルは `Nq == 8`（p=7）と
   `Nq == 256`（p=255）にしか存在せず、それ以外は
   `error stop "CUDAFORTRAN_FUSED requires Nq=8 or Nq=256"` になる
   （`mod_advect3d_eq.f90:605,613,622` および `:652,660`）。
   p=15/31/63/127 で走るのは汎用の OpenACC パスだけなので、測っても
   「要素並列カーネル vs GEMM 化」の比較にはならず、汎用パス同士の比較にしか
   ならない。損益分岐を知りたいなら、まず中間次数の専用カーネルを書く
   必要がある。それは測定ではなく実装の作業であり、現時点で優先度は低い。
5. ~~**1 step あたりの launch 回数の削減（CUDA Graph）。**~~ → **完了
   （2026-08-25、§8.3）**。1 step を捕捉して再生するようにし、p=7 `FUSED_TC` の
   Main は 1.2038 → 1.1716 秒（−32 µs/step）、p=255 `GEMM_FUSED` は
   3.9730 → 3.8545 秒（−119 µs/step）。namelist の `UseCudaGraph` で選ぶ。
   再生時は tendency の CUDA event 時間が採れないので、device 時間を見たい
   測定では off にすること。
6. ~~**p=255 の lift の epilogue 融合。**~~ → **完了（2026-08-25、§8.4 / §8.5）**。
   3 本の `K=2` GEMM → 1 本のカーネル → z-epilogue 内で 6 面から直接評価、
   の順に `lift_out` を消した。`GEMM_FUSED` の Main は 3.971 → 3.463 秒
   （−12.8%）で、旧実装とビット一致する。残る独立カーネルは
   `volume_flux_kernel`（150 µs、DRAM 66%）で、`q*vel` の mainloop 融合は
   やり直さない。→ その `volume_flux_kernel` も §8.6 で 125.9 µs / DRAM 83.4% に
   なった（ロードをストアより前にまとめただけ。トラフィックは元から理論最小で、
   実体は帯域律速ではなくレイテンシ律速だった）。
   ここで得た一般則: **epilogue に足した整数演算は、mainloop が SM
   throughput 律速のとき end-to-end にそのまま出る。** 素直に書いた版
   （出力 1 点ごとに `p % Nq` / `p / Nq`）は −0.6%、CUTLASS の
   `operator++` が row しか進めないことを使って column 不変量を
   ループ外に括り出した版が −4.7% である。
7. ~~**帯域律速カーネルを GEMM の裏に隠す（2 本目のストリーム）。**~~ →
   **完了（2026-08-25、§8.7）**。隠せたのは `elembnd_flux_kernel`（19.6 µs）
   だけで、volume flux の分割は不採用。graph replay では損になるので graph
   モードでは無効化している。Main は `GEMM_FUSED`（graph off）3.3723 →
   3.3293 秒。以下は着手前の見立てである。§8.6 の
   再測定で、x GEMM（249.7 µs）と y GEMM（243.5 µs）は SM 88–89% で回りながら
   DRAM を 10.1% / 6.3% しか使っておらず、SM あたりのレジスタも 2 CTA ×
   27,136 = 54,272 / 65,536 で約 11k 本空いている。x GEMM が要るのは `flux_x`
   だけ、y GEMM が要るのは `flux_y` だけ、z GEMM が要るのは `flux_z` と
   `flux_bnd` だけなので、`flux_y`/`flux_z` と `elembnd_flux_kernel`（19.6 µs）を
   2 本目のストリームに逃がして event で join すれば、クリティカルパスから
   約 110 µs / call（tendency 978 µs の 11%）を外せる可能性がある。
   カーネル本体も演算順序も変わらないのでビット一致が期待でき、CUDA Graph
   捕捉も fork/join を event で書けば通る。代償は `q` を 2 回読むこと
   （+134 MB、ただし隠れる側）と、GEMM 側がどれだけ遅くなるかで、そこは
   実測でしか決まらない。
8. **全パスの point-varying 係数回帰の自動化**（CI 化）。

最優先は性能ではなく、`D(q*velocity)` と 6 面数値流束という
元実装の意味を守り続けることである。

### 12.9 補足: TMA（Tensor Memory Accelerator）は使えるか（調査メモ、2026-08-25）

結論から言うと**使えるが、既製品は無い**。CUTLASS のテンプレート引数を
`arch::Sm80` から `Sm90` / `Sm100` に替えれば TMA になる、という話ではない。

| 項目 | 状況 |
|---|---|
| ハードウェア / toolchain | GB200 = sm_100、`nvcc 13.1`。TMA は sm_90 以降なので利用可 |
| FP64 データ型 | **対応**。`third_party/cutlass/include/cute/arch/copy_sm90_desc.hpp:220` に `is_same_v<T,double> -> CU_TENSOR_MAP_DATA_TYPE_FLOAT64` があり、`make_tma_copy` は double で通る |
| CUTLASS の collective builder | **FP64 の特殊化が 1 つも無い**（`include/cutlass/gemm/collective/builders/` を `double` で grep して 0 件）。SM90 の builder は wgmma 前提で、**wgmma に FP64 は無い** |
| FP64 tensor core 命令 | 今も `mma.sync` 系（DMMA）。CuTe には `MMA_16x8x{4,8,16}_F64F64F64F64_TN`（`include/cute/arch/mma_sm90.hpp:52,85,118`）がある |

TMA はコピーエンジンであって MMA とは直交しているので、
**TMA で global→shared を運び DMMA で回す mainloop は原理的に書ける**。
ただし CUTLASS 4.7 にその組み合わせは無く、`CollectiveMma` の手書きになる。
傍証として、GB200 上で cuBLAS 自身が DGEMM に投げてくるのは現在も
`cutlass_80_tensorop_d884gemm_*`、すなわち cp.async 世代の SM80 カーネルである
（§5.1）。NVIDIA も FP64 に TMA 版を出していない。

期待値は、§8.6 で採り直した時間を §7 の理論 FLOP で割ると見積もれる。

| kernel | nsys | 実効 FP64 | 対 40.1 TFLOP/s |
|---|---:|---:|---:|
| x GEMM | 249.7 µs | 34.4 TFLOP/s | **85.8%** |
| y GEMM | 243.5 µs | 35.3 TFLOP/s | **88.0%** |
| z GEMM（assembly + lift epilogue 込み） | 339.1 µs | 25.3 TFLOP/s | 63.2% |
| volume 3 本合計 | 832.3 µs | 31.0 TFLOP/s | 77.3% |

- **x/y GEMM に入れる意味はほぼ無い。** 既に FP64 ピークの 86–88% で、GB200 では
  TC ピーク = CUDA core ピークだから TMA は天井を上げない（§7）。
- **z GEMM には一応の筋がある。** 63.2%、レジスタ 254 本、occupancy 12.2% で、
  lift を epilogue に入れた代償として SM throughput が 79.8 → 72.5% に落ちている
  （§8.6）。SM80 multistage mainloop は `PredicatedTileIterator` のアドレス状態と
  cp.async のポインタをレジスタに持つが、**TMA はディスクリプタ駆動でレジスタを
  ほぼ消費しない**。epilogue に押されているレジスタを mainloop 側から返せる、
  というのがこの方向の唯一の具体的な狙いである。
- **p=7 `FUSED_TC`。** `tc_paper_survey` §12.4 の不採用理由は「レジスタ 32 本
  ちょうどで先行ロードを保持する余地が構造的に無い」だった。TMA は
  レジスタを使わない先行ロードで、データ経路も L2→SMEM で L1/TEX を通らない
  （同 §12.5 は残る律速を「L1/TEX 90% 張り付き」と結論づけている）。ただし
  TMA は矩形タイルしか運べないので `VMapM`/`VMapP` の face gather は対象外で、
  TMA 化できるのは volume の `q,u,v,w` だけ。加えて smem が 28.2 KB × 8 ブロック
  = 225 KB で 228 KB をほぼ使い切っており、ステージングバッファは occupancy を
  削る。CUDA core 版の `FUSED` は 15.9 KB × 8 = 127 KB で余裕があるが、そちらは
  CUDA Fortran で、**nvfortran は TMA を公開していない**ので `.cu` への移植が要る。

実務上の条件: `cuTensorMapEncodeTiled` は global ベースアドレス 16 B 境界と
最内 stride 16 B 倍数を要求する。`flux_x`(lda=256→2048 B)、`flux_y`(256)、
`flux_z`(ld=65536) はいずれも満たす。ディスクリプタは 1 回作って使い回せる。
未確認なのは、8 B 要素で CUTLASS の 128B swizzle atom がそのまま噛むかどうか。

優先度としては項目 7（2 本目のストリームによる重ね合わせ、約 110 µs/call）の
ほうが測定で裏が取れている。TMA を試すなら z GEMM に絞り、先に
`ncu --set full` でレジスタ起因の stall を確認してから mainloop を書くこと。

### 12.10 追記: 上の §12.9 を実測した結果、採用はゼロ（2026-08-26）

§12.9 の最後の指示（「先に `ncu --set full` でレジスタ起因の stall を
確認してから」）に従って実測した。**確認したら、狙いのほうが消えた。**
詳細は `tma_survey.md`。以下は §12.9 の記述のうち訂正が要る箇所だけを挙げる。
上の表と本文は 2026-08-25 時点の記録としてそのまま残す。

1. **「狙えるとすれば z GEMM のレジスタ圧だけ」は成立しない。**
   z GEMM は融合版・CUTE 版のどちらも `Block Limit Shared Mem` が **4** で、
   CUTE 版はレジスタ 156 本（`Block Limit Registers` = 6）でも 4 ブロック/SM に
   張り付いている。律速は dynamic shared memory 49.15 KB のほうなので、
   **レジスタを mainloop から返しても occupancy は 12.5% から動かない**。
   さらに、TMA が隠せる long scoreboard stall は融合版でも **11.33%** しかなく
   （固定レイテンシ依存 36.64%、演算パイプ 20.44%）、融合 epilogue の増分時間の
   実体は 5 本の追加オペランドの DRAM **701.7 MB/call**（理論 671.1 MB）である。
   これはどのコピー機構でも減らない。

2. **「p=7 は smem 28.2 KB × 8 = 225 KB でほぼ使い切っており余地が無い」は
   `e22dda1` 以前の値である。** 現行の静的 smem は 16,256 B で、
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` の実測では **+8,192 B まで
   8 ブロック/SM を保てる**。ただし p=7 が不採用になった理由は予算ではなく
   **レイアウト**で、TMA が 8 B 要素で符号化できる 4 通りのレイアウト全部で
   x/y 収縮が 2-way バンクコンフリクトになる（`tma_survey.md` §3.2）。

3. **「nvfortran は TMA を公開していないので `.cu` への移植が要る」は
   `FUSED_TC` には当てはまらない**（最初から `.cu`）。CUDA core 版 `FUSED` には
   当てはまる。

4. **「8 B 要素で 128B swizzle atom が噛むかは未確認」→ 噛む。**
   ただし swizzle を付けると最内 box 次元が 32B → 4、64B → 8、128B → 16 doubles に
   固定される。加えて box の各次元は 256 要素以下、global ベースは 16 B 境界、
   `globalStrides` は 16 B の倍数が必要である。

5. **前提として、現行のビルドフラグでは TMA が有効になっていない。**
   `Makefile:11` の `-arch=native` は sm_100 に解決され、CuTe の
   `CUTE_ARCH_TMA_SM90_ENABLED` は sm_100 では定義されない（`sm_100a` / `sm_100f`
   が要る）。

6. **付随して見つかった、TMA より期待値の高い標的。** z GEMM は融合版・CUTE 版
   ともに shared store が平均 8.3–8.5-way のバンクコンフリクトを起こしており、
   全 store wavefronts の **52%**、ncu の推定改善余地は **28.8–30.5%** である。
   両版で同値なので自作 epilogue ではなく **CUTLASS 標準の `MmaMultistage`** 側に
   ある。p=255 最速パスの単独最大カーネル（nsys 340.0 µs、tendency の 31.6%）
   なので、§12 の次の標的はここにすべきである。

---

## 13. 関連ファイル

| ファイル | 内容 |
|---|---|
| `AGENTS.md` / `CLAUDE.md` | 数値契約、ビルド、検証、プロファイル、コミット方針 |
| `README.md` | 実装パスの説明と実行方法 |
| `execution_times.md` | `nstep=1000` の同一条件パス別実行時間（job 41348、`e22dda1` の追記あり） |
| `gpu_optimization_session_report.md` | GPU 対応・最適化の経緯と失敗の記録 |
| `p255_gemm_fusion_session_report.md` | p=255 GEMM / epilogue 融合の詳細実験 |
| `tc_paper_survey_2407.09621.md` | arXiv:2407.09621 の調査と p=7 Tensor Core カーネルの shared レイアウト刷新 |
| `bench_runs/` | 比較用 input・job・ログ（job 41348） |
| `output/` | nsys / ncu レポートと run log（job 43219） |
| `job_all.sh` / `gen_ncu_cmds.sh` | 全パス一括 run + nsys + ncu 採取 |
| `main.f90` | 時間積分ループ（`main_64`=q0 コピー, `main_87`=RK 更新, `main_99`=min/max） |
| `mod_advect3d_eq.f90` | tendency dispatch と作業配列 |
| `mod_cuda_dg_kernels.cuf` | CUDA Fortran 実装（SPLIT / FUSED / GEMM 系） |
| `cuda_dg_kernels_tc.cu` | FP64 Tensor Core `m8n8k4` 実装 |
| `cuda_cublas_gemm.cu` / `cuda_cutlass_gemm_fused.cu` | cuBLAS / CUTLASS GEMM 経路 |
| `cutlass_z_gemm_assembly.h` | z batched GEMM の TensorOp mainloop + assembly epilogue |
| `mod_mesh.f90` | mesh・マッピング・halo・p=255 演算子生成 |

---

## 13. 追記: 律速の棚卸しと、そこから出た 2 件（2026-08-26、`63a4234`）

「各カーネルが何に律速され、ハードウエア上限に対してどこまで出ているか」を
両次数の最速パスについて採り直した（Slurm job `49566` / `49567`）。
`ncu --set full` は SM クロックを 1.08 GHz に固定するので、**DRAM 比率は
`ncu 時間 / nsys 時間` で実運用クロックに換算**してある（SM・L1 の
throughput 比はクロックに依存しないのでそのまま）。換算値は §7.1 と一致する。

### 13.1 カーネル別の律速（HEAD `63a4234`）

以下は**着手前**（`63a4234`）の棚卸しである。§13.2 を入れた後の値は §7 / §7.1
に反映してある。

**p=7 `CUDAFORTRAN_FUSED_TC`, Ne=32³（1 step = 1073 µs）**

| kernel | µs/step | 律速 | 達成率 | 余地 |
|---|---:|---|---|---|
| `tendency_fused_p7_tc_kernel` ×3 | 834 (77.7%) | **L1/TEX 91.8%** | DRAM 72.4% / FLOP 12.5% / occ 96.4% | あり（§13.2）|
| RK 更新 stage 1 / stage≥2 | 224 (20.9%) | **DRAM 79–87%** | SM 35.5% | 転送量以外に無し |
| `update_halo` ×3 | 15 (1.4%) | レイテンシ（1.26 wave）| DRAM 33% | 絶対値が小さい |

**p=255 `CUDAFORTRAN_GEMM_FUSED`, Ne=1（1 step = 3222 µs、クリティカルパス 3141 µs）**

| kernel | µs/step | 律速 | 達成率 | 余地 |
|---|---:|---|---|---|
| z GEMM `<64,32,16>`（assembly + lift epilogue）×3 | 1017 | SM 73.1% | FP64 65% / DRAM 34.8% / occ 12.5%（smem 4 block）| 構造的に小さい |
| x GEMM `<64,128,16>` ×3 | 764 | **SM 87.9%** | FP64 84% / DRAM 18.4% | 打ち止め |
| y GEMM `<64,64,16>` ×3 | 729 | **SM 89.6%** | FP64 88% / DRAM 11.8% | 打ち止め |
| `volume_flux_kernel` ×3 | 387 | **DRAM 87.4%** | 転送量は理論最小の 1.00 倍 | 打ち止め |
| RK 更新 | 225 | **DRAM 79–87%** | — | §13.3 で否定 |
| `elembnd_flux_kernel` ×3 | (81) | — | x GEMM の裏で完全に隠れる | コストゼロ |
| `update_halo` ×3 | 18 | レイテンシ | DRAM 42% | 小さい |

重要な訂正: §8.8 が RK 更新を「DRAM 74.2%」と書いていたのは **ncu の値を
クロック換算せずに読んだもの**である。実運用クロックでは 79–87% で、
HBM3e の実効としてはほぼ上限であり、削り代は転送量にしか無い。

### 13.2 p=7: 無駄は 6 面のうち 2 面に全部あった（採用、−5.1%）

p=7 の DRAM 転送は理論最小の **1.02 倍**（実測 1.469 GB / 理論 1.44 GB）で
無駄が無い一方、**L1 のロードセクタはユニークバイト数の 2.09 倍**である。
`-lineinfo` 付きで Source ページを採ると（job `49589`）、超過 2477 万セクタは
**±x 面（面 2 と 4）の gather 8 本**に全部乗っていた。`Fmask` が ±x 面に
`i0 + 8j + 64k` を与えるため、warp の 32 レーンが 8 doubles 飛びになり、
32 B セクタの 8 B しか使わない。±y / ∓z 面は連続で理想値ちょうどである。

対策は、その 2 面の **M 側だけ**を shared 経由にすること。M 側は同じ要素の
`i = 0` / `i = 7` 面で、volume セクションが coalesce したロードで既に
レジスタに持っている値である。4.6 KB のステージングバッファを足し、
face セクションでは `fp & 64` で warp ごと分岐を揃えて読む。

| | ベース | 採用版 |
|---|---:|---:|
| global load sectors | 95,944,704 | **78,643,200**（−18%）|
| long scoreboard stall | 35.07 | **24.03** |
| ncu 1 launch | 403.6 µs | 388.4 µs |
| `CUDA device fused tendency`（`nstep=1000`）| 0.8497 s | **0.8060 s（−5.1%）**|
| Main（graph off / on） | 1.1092 / 1.0738 | **1.0656 / 1.0367** |

`63a4234` と**ビット一致**。詳細と、途中で外した 3 版は
`tc_paper_survey_2407.09621.md` §13。

そこで一番高くついた誤りを一般則として残す。**shared を増やすとき
`cudaSharedmemCarveoutMaxShared` を反射的に足さないこと。** 既定の carveout で
ブロック数が足りるかは `launch__occupancy_limit_shared_mem` で 1 回測れば
分かる。carveout を最大にすると L1 データキャッシュが削られ、L1 が最繁
ユニットのカーネルでは **+31%** になった。占有率は 1 ブロックも増えていない。

### 13.3 p=255: RK 更新を z epilogue に融合するのは損（不採用）

§13.1 の RK 更新（225 µs/step、DRAM 79–87%）を速くする唯一の道は転送量を
減らすことで、`dqdt` を作っているのは z GEMM の epilogue だから、そこで
`q = a*q0 + b*(q + dt*dqdt)` まで済ませれば `dqdt` のストア 134 MB と
ロード 134 MB がステージごとに消える。z GEMM は DRAM 34.8% と帯域に余裕が
あり、`elembnd_flux_kernel`（`q` を読む唯一の同時実行カーネル）は
z GEMM の前で join 済みなので、`q` の in-place 更新も安全である。

実装して測った（`FuseRKUpdate` を namelist に足し、epilogue を
`kRkMode` 0/1/2 でインスタンス化。0 は従来、2 が stage 1 で `q0 <- q` も行う）。
Slurm job `49674`、`nstep=20`。

| kernel | 融合なし | 融合あり |
|---|---:|---:|
| z GEMM（stage≥2） | 338.8 µs | **481.1 µs（+42%）**|
| z GEMM（stage 1、`q0` 保存込み） | 338.8 µs | 452.0 µs（+33%）|
| RK 更新カーネル | 76.6 + 2×74.5 µs | **削除** |
| 1 step の z + RK | 1241.9 µs | 1414.2 µs |
| Main（`nstep=1000`、3 ラウンド）| **3.228 s** | 3.407 s（**+5.5%**）|

**オペランドを 3 本足すと 1 call あたり 142 µs 増える**。追加の DRAM は
402 MB/call で、これを 481 µs で流しても DRAM は 35 → 42% にしかならない。
つまり帯域ではなく、**4 ブロック/SM（shared 律速）では隠せないロード
レイテンシ**が実体である。stage 1（読み 1 本・書き 2 本）が stage≥2
（読み 2 本・書き 1 本）より 29 µs 安いことがその裏づけで、効いているのは
バイト数ではなく読みの依存段数のほうである。`tma_survey.md` §2 が
「増分時間の実体は 5 本の追加オペランドの 701.7 MB/call」と書いたのと同じ壁で、
今回は 3 本で同じ値段（1 本あたり 35–47 µs/call）が付いた。

一般則 13（速くなったものだけ残す）に従って**コードは差し戻した**。

**p=7 側で同じ融合をやる案は、性能以前に成立しない。**
`tendency_fused_p7_tc_kernel` は P 側 gather で `q[iP]`、すなわち**隣接要素の
`q`** を読む。epilogue で `q` を in-place 更新すると、あるブロックが要素 A の
`q` を書いている最中に別のブロックが同じ `q` を face 点として読むので、
ブロック間の read-after-write レースになる。ping-pong バッファにすれば
避けられるが、SSP-RK3 は 3 ステージ（奇数）なので 1 step で役割が入れ替わり、
CUDA graph に焼いたポインタが再生時に合わなくなる。

## 14. 経路横断の再測定（2026-08-29）

本レポート本文の時間は 2026-08-25 前後のスナップショットである。表は書き換えない。
p=7…255 を現行ツリー（commit `2dadc41`）で採り直した結果は
[`README.md`](README.md) の「最新結果のまとめ」と、各次数レポート
（p=7 は `execution_times.md` 追記 16、p=15 §18、p=31 §16、p=63 §22、
p=127 §15、p=255 §12）にある。最速パスの順位は変わっていない。
