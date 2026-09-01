# p=7 `CUDAFORTRAN_GEMM_FUSED`

GB200 1 GPU、`make CUDA=1 GPUFLAGS=-gpu=cc100`（nvcc `-arch=sm_100`）、
`namelists/perf_p7_gemm_fused.conf`（`Ne=32³`、`nstep=20`、graph off）。
親 commit `40fff1a` の作業ツリー。対象は `CUDAFORTRAN_GEMM_FUSED`。最速は
`FUSED_TC`（274.9 µs/stage）のまま。この次数で GEMM 経路が最速を抜く見込みは
無い（同一 DOF で面点率 75%、経路 DRAM は unique の 2.6 倍）。作業の目的は
`Nq*Ne > 65535` で閉じていた列を開き、浅い `K=8` で融合が負ける理由を測って
モデルに残すことである。

**2026-08-31 訂正**: §6--9 の 510--380 µs は要素 CTA の手書き CUDA-core
体積カーネルへ置換した探索値であり、`GEMM_FUSED` の役割（`GEMM_CUTE` と
CUTLASS mainloop を共有）を満たさない。測定済みの範囲外結果として表は残すが、
本番コードには採用しない。役割準拠の最終値は §10 の device **1960.3
µs/stage**、Main **6.042 ms/step** である。

µs/stage = `CUDA device GEMM fused` / (19 measured steps × 3 RK)。
点変化係数の owned `dqdt` は `Ne=2³` で `CUDAFORTRAN_SPLIT` と最大絶対差 0、
`Ne=32³`（16,777,216 点）で `CUDAFORTRAN_GEMM` と**ビット一致**（§2 まで。
§6 の要素 CTA は加算順が変わりうるが、SPLIT との最大絶対差は 0）。

現在の占有 GPU interleaved A/B は device **1960.3 µs/stage**、Main
**6.042 ms/step**（§10）。`FUSED_TC` の **7.13 倍**。

nsys: Slurm job `70974`（c082、cuBLAS y 時代）、`70989`（手書き y/z）、
`70991`、`71056`（flux_deriv）。`--resolve-symbols=false`、
`DEBUGINFOD_URLS=`。login 5-run 中央値で採否した（占有 GPU の 12 回交互は
未実施。差が数 % を超える採用はレンジ非重複）。

## 0. なぜ無かったか

y GEMM は `Nq*Ne` 平面の CUTLASS `GemmBatched` で、batch が `grid.z` に載る。
CUDA の `grid.z` 上限は 65535。p=7 `Ne=32³` は `8*32768 = 262144`。
ゲートは `mod_advect3d_eq.f90` と `cuda_cal_dqdt_gemm_cutlass` の両方にあった。

## 1. 有効化: batch を 65535 で分割

`cuda_cutlass_gemm_fused.cu` の batched 起動（y、scaled y、y-scaleadd、z
assembly）を `kCudaMaxGridZ` ずつに切る。タイル型は変えない。p=15
（`Nq*Ne = 65536`）も同じ経路で通る。

これだけで走らせた login 5-run 中央値は device **3678 µs/stage**、Main
**11.11 ms/step**。同一実行ファイルの `CUDAFORTRAN_GEMM` は **1634 µs** /
**5.08 ms**。CUTLASS `GemmY` は 64×64 タイルで、8×8 問題を CTA 1 個に載せる。

## 2. 採用

| 変更 | device s（19 step） | µs/stage | vs 直前 |
|---|---:|---:|---|
| batch 分割のみ | 0.20966 | 3678 | — |
| `Nq<64` の y を cuBLAS strided-batched | 0.12704 | 2229 | **−39.4%** |
| `Nq<64` の z を cuBLAS + `separable_lift_assembly` | 0.09552 | 1676 | **−24.8%** |
| elembnd を volume GEMM の裏に置かない | 0.09360 | 1642 | **−2.0%** |
| y を side2 で cuBLAS x と重ねる | 0.09261 | 1625 | **−1.1%** |
| z を idle の side stream で x/y と重ねる | 0.09239 | **1621** | **−0.23%** |

最終 login 5-run 中央値（当時）: Main **5.045 ms/step**、device **1621 µs/stage**。
0.867 TFLOP/s（ピーク 2.2%）、unique DRAM 0.97 TB/s（12.3%）。
`CUDAFORTRAN_GEMM` の 1634 µs に対し **−0.8%**（login）。`FUSED_TC` の
**5.9 倍**遅い。

cuBLAS 呼び出しは `dg_cuda_stream` を毎回 `cublasSetStream` する。side2 に
載せた y が本流の x と同じハンドルに落ちないようにするためで、未融合 GEMM
の既定ストリームでは no-op である。

1 文: **浅い `K=8` では CUTLASS の 64 幅タイルと z 融合エピローグがどちらも
負け、残る仕事の半分は cuBLAS y である。**

## 3. nsys（job `70974`、最終形の直前 = x\|\|y、z は直列）

20 step × 3 RK = 60 stage（warmup 込み）。カーネル平均:

| カーネル | 平均 | 1 stage の本数 | 1 stage |
|---|---:|---:|---:|
| cuBLAS y `d884gemm_32x32` | 161 µs（中央 200、短いprologue 2.7） | 5 | **~800 µs** |
| cuBLAS x `d884gemm_32x64` | 62 µs | 3 | **~185 µs** |
| `elembnd_flux_kernel` | 169 µs | 1 | 169 |
| `separable_lift_assembly_kernel` | 167 µs | 1 | 167 |
| cuBLAS z `d884gemm_64x32` | 161 µs | 1 | 161 |
| `volume_flux_kernel` | 126 µs | 1 | 126 |

y が tendency の約半分。x を y の裏に置くと 185 µs が隠れ、実測 −1.1% と一致する。
z をさらに重ねても y が DRAM を張っているので −0.23% しか出ない。

## 4. 不採用

| 候補 | 結果 | 理由 |
|---|---|---|
| CUTLASS z assembly（64×32 タイル、融合エピローグ） | 単独 **876 µs/stage**（job `70970`） | `K=8` の `(Nq²×Nq)×Ne` に対してタイルが過大。cuBLAS z + 既存 assembly が床 |
| elembnd を x/y の裏へ（cuBLAS y のあと再試行） | **+3.2%** | 歴史的な GEMM の +3.1% と同じ。両方 DRAM 律速で互いに直列化する。窓が 876 µs から 800 µs に変わっても符号は変わらない |

範囲外: 速度の事前計算、`dqdt` を実体化しない RK 融合、代表スカラー特殊化。

## 5. 残りの天井と終了（当時。§6 で覆った）

契約内で残る塊、と当時書いたもの:

- **cuBLAS y（~800 µs、~49%）**。算術屋根 ~13 µs、2 本の体積テンソルの
  DRAM ~34 µs。ライブラリの 32×32 タイルが床ではなかった。
- **elembnd 169 µs / assembly 167 µs / flux 126 µs**。

§5 の「探索終了」は、y の天井を測らずに書いたので誤りである。

## 6. 浅い K をライブラリから外す（login 5-run 中央値）

点変化係数の owned `dqdt` は `Ne=2³` で SPLIT と最大絶対差 0 のまま。

| 変更 | device s（19 step） | µs/stage | vs 直前 |
|---|---:|---:|---|
| §2 末（cuBLAS y/z、x\|\|y\|\|z） | 0.09239 | 1621 | — |
| Nq=8 手書き y/z（512 thr/elem、shared D） | 0.04639 | 814 | **−49.8%** |
| `separable_lift_assembly_p7`（512 thr/elem） | 0.04562 | 800 | −1.7% |
| 手書き x（8×64 または 8×8 タイル） | 0.094 | ~1649 | **+106%（不採用）** |
| cuBLAS x を strided-batched 8×64×Ne | 0.04529 | 794 | −0.6% |
| `volume_deriv_p7`（3 収縮を 1 CTA、shared flux） | 0.03775 | 662 | **−16.6%** |
| `volume_flux_deriv_p7`（flux を shared に閉じる） | 0.02907 | **510** | **−23.0%** |
| elembnd を flux_deriv の裏へ | 0.02888 | 507 | −0.7%（DRAM 同士、nsys は和≈壁。不採用） |

最終採用は **flux+deriv 融合、elembnd 直列**。login 5-run: Main **1.765 ms/step**、
device **510 µs/stage**。`FUSED_TC`（274.9）の **1.85 倍**。旧 1621 の **3.18 倍**速い。

nsys job `70989`（手書き y/z、generic assembly）: y 79、z 79、cuBLAS x 91×2=182、
elembnd 169、assembly 167、flux 126。y は 800→79 で、DRAM 屋根 34 の 2.3 倍。

nsys job `70991`（y/z + p7 assembly + strided x）: x は 1 ローンチ 180 µs のまま
（32×64 タイル）、assembly p7 **151 µs**、y 79、z 78。

nsys job `71056`（flux_deriv + elembnd 重ね）: **3 本** — elembnd 177、
`volume_flux_deriv_p7` 172、assembly p7 146。カーネル和 ≈ 壁時間。2 本の
DRAM カーネルを重ねても直列化する。

### 不採用（§6）

| 候補 | 結果 | 理由 |
|---|---|---|
| 手書き x（左乗算 C=D*A） | +106% | 列方向 K ロードが非 coalesced。8×8 タイル+smem 転置は同期が増えてさらに遅い |
| elembnd \|\| flux（join を GEMM 前） | login ノイズ、中央値は悪化側 | 両方 q,u,v,w を読む |
| elembnd \|\| flux_deriv | nsys 和が壁に一致 | 両方 DRAM。歴史的 GEMM +3.2% と同符号 |
| `volume_flux_deriv_lift_assembly_p7` | 未実装 | `Lift_mat` 密行列。経路を `FUSED` に置き換える |

1 文: **浅い K=8 の「GEMM」はライブラリではなく要素 CTA の 3 収縮であり、
残る壁は面 gather と assembly の 2 本の帯域カーネルである。**

## 7. assembly を体積 CTA に畳む

点変化係数の owned `dqdt` は SPLIT と最大絶対差 0。

| 変更 | device s（19 step） | µs/stage | vs 直前 |
|---|---:|---:|---|
| §6 末（flux_deriv + assembly 分離） | 0.02907 | 510 | — |
| `volume_flux_deriv_assembly_p7`（Lift1D、deriv 非書き） | 0.02300 | 403 | **−21.0%** |
| elembnd 384 thr/elem | 0.02274 | **399** | −1.1% |
| elembnd M 側を shared 体積に載せる | 0.02392 | 420 | **+5.2%（不採用）** |

採用は **fused volume+separable assembly + 384-thread elembnd**。login 5-run:
Main **1.438 ms/step**、device **399 µs/stage**。`FUSED_TC` の **1.45 倍**。

nsys job `71059`（fused assembly、generic elembnd）: volume+assembly **216 µs**、
elembnd 169。和 385、壁 403。

ncu job `71060`（`--set full`、510 µs バイナリ、nstep=4。この ncu は SOL 91
メトリクスのみで stall 内訳は無い）:

| カーネル | DRAM | L1/TEX | 占有率 | L1 hit |
|---|---:|---:|---:|---:|
| elembnd | 69% | 89% | 86% | **5.8%** |
| flux_deriv | 41% | 91% | **47%** | 3.2% |
| assembly p7 | 76% | 81% | 85% | 36% |

elembnd は L1 上の gather（ヒット率 6%）。flux_deriv は shared で占有率が半分。
assembly は DRAM。ncu 時間はクロック固定のため壁時間の採否には使わない。

### 不採用（§7）

| 候補 | 結果 | 理由 |
|---|---|---|
| `Lift_mat` 版 `volume_flux_deriv_lift_assembly_p7` | 未使用 | 密 lift。経路を `FUSED` に置き換える |
| elembnd の M 側 shared | +5.2% | 要素全体 4 場を読んで P 側 gather は残る。バリアの方が高い |

契約内で残る塊:

- **elembnd ~169 µs（nsys、~42% of 399）**。P 側 gather は隣接要素。速度の
  事前計算は範囲外。M 側 shared は負け。
- **volume+assembly ~216 µs**。q,u,v,w と面、Escale、dqdt。elembnd と畳む
  のは `FUSED` の仕事。
- stall メトリクスの ncu と占有 GPU 12 回交互は未。

## 8. 天井のアブレーションと面 shared

`SCALE_DG_SKIP`（login 3-run 中央値、getenv 導入後のベース 399 µs）:

| 消したもの | device s | µs/stage | 消えた分 |
|---|---:|---:|---:|
| ベース | 0.02274 | 399 | — |
| elembnd | 0.01326 | 233 | **166 µs** |
| 体積+assembly | 0.01027 | 180 | **219 µs** |

elembnd 166 µs は nsys 169 と一致する。P 側 gather を消すのは範囲外。
体積 219 µs に対し unique 転送の DRAM 屋根は約 148 µs（約 **70 µs** が契約内の上限）。

ncu stall（job `71074`、399 µs バイナリ、クロック固定）:

- elembnd: **long_scoreboard 35.3**、`lg_throttle` 12.6。sector/request **11.9**。
  L1 gather 待ち。
- 体積+assembly: long_scoreboard 6.2、mio 4.1。shared ld コンフリクト 254 k。

| 変更 | device s | µs/stage | vs 399 |
|---|---:|---:|---|
| flux_bnd と Lift1D を体積 CTA の shared へ | 0.02257 | **396** | −0.7% |

login 5-run はベースとレンジ非重複。占有 GPU の 12 回は未。SPLIT 差 0。

契約内で残る塊:

- **elembnd 166 µs** の仕事自体は契約。残る機構は gather レイテンシ
  （long_scoreboard）。M 側 shared は §7 で負け。速度の事前計算は範囲外。
- **体積 ~70 µs** が DRAM 屋根まで。shared 面で 3 µs。バンクコンフリクトは
  残っている。
- 2 本を 1 本にするのは `FUSED`。

## 9. 体積 shared の j パディング

ncu job `71074` の体積+assembly は shared ld コンフリクト 254 k。1D
`sflux_{x,y,z}(512)` は y/z の内積が先頭次元 8 のストライドになる。
`sflux_{x,y,z}(8,9,8)` に直し、ストア/ロードを `(i,j,k)` にした（p=63 の
エピローグと同じ「遅い添字を 9」）。z の 2 平面は従来どおり同じ `l` 線を共有。

login 5-run（レンジ非重複、GPU が静かなときの測定）:

| 変更 | device s | µs/stage | Main ms/step | vs 396 |
|---|---:|---:|---:|---:|
| `sflux_*` を `(8,9,8)` | 0.02167 | **380** | 1.382 | **−4.0%** |
| `sflux_y` のみ `(9,8,8)` | 0.02166 | **380** | 1.382 | 0（y 内積の `l` ストライド 9。壁は同じ） |

ncu job `71130`（380 µs バイナリ、クロック固定、3 回の中央）: 体積の shared ld
コンフリクトは 254 k → **~410 k** に増えた。long_scoreboard は 6.2 → **3.45**、
mio は 4.1 → **7.0**。壁時間の −4% はコンフリクト削減ではなく、パッド後の
LDS パターンが DRAM 待ちを隠したため。j を 9 にしても Fortran 列優先では
`sflux_y(i,l,k)` の `l` ストライドは先頭次元 8 のままなので、y 内積のバンクは
直っていない。

SPLIT 差 0。`SCALE_DG_SKIP`（3-run 中央、ベースを 380 とする）:

| 消したもの | device s | µs/stage | 消えた分 |
|---|---:|---:|---:|
| elembnd | 0.01221 | 214 | **166 µs**（§8 と同じ） |
| 体積+assembly | 0.01037 | 182 | **198 µs**（§8 の 219 から −21 µs） |

DRAM 屋根 ~148 µs に対し体積はまだ約 **50 µs**。

### 不採用（§9）

| 候補 | 結果 | 理由 |
|---|---|---|
| P 側 `q,u,v,w` の `__ldg`（C++ 384-thread elembnd） | ~380 µs（レンジ重複） | `FUSED` では効いたが、ここは CTA に M 側 shared が無く gather が L1 をほぼ外す（ヒット 5.8%）。テクスチャ経路に載せても待ちは残る |
| `sflux_bnd` を `(8,9,6)` | **403 µs（+6%）** | 面の 2 添字は面ごとに (i,k)/(j,k)/(i,j)。パッドした遅い添字がロードと一致しない。索引計算が増える |
| Fortran `launch_bounds(256,4)` | 未測 | nvfortran は `attributes(global, launch_bounds(...))` を構文エラーにする |

契約内で残る塊:

- **elembnd 166 µs**。`__ldg` は負け。M 側 shared は §7 で負け。P 側を
  消すのは範囲外。
- **体積 ~50 µs** が DRAM 屋根まで。j パッドで 21 µs（ncu ではコンフリクトは増えた）。
  `sflux_y(9,8,8)` は壁を動かさない。面配列パッドは負け。
- 2 本を 1 本にするのは `FUSED`。

## 10. 役割準拠の CUTLASS mainloop 最適化（2026-08-31）

§6--9 の手書き CUDA-core 体積カーネルを撤回し、`GEMM_CUTE` と
`GEMM_FUSED` が次を共有する構成へ戻した。

- `Nq<=64` の x は共通 helper から同じ wide cuBLAS GEMM。
- y/z は `VolumeGemmSetP7` の同じ CUTLASS tile。CUTE は非加重 epilogue +
  separate assembly、FUSED は z assembly epilogue だけが異なる。
- `grid.z=65535` を超える batch は tile を変えず複数 launch に分割する。
- p=7 FUSED だけは独立な x/y を別 stream で重ねる。これは fusion package の
  schedule で、mainloop 自体は同一である。

最終 tile は y `TB=32x64x16 / warp=32x32x16 / stages=3`、z
`TB=16x32x16 / warp=16x16x16 / stages=3`。すべて FP64 TensorOp
`mma.sync.m8n8k4` である。

### 10.1 採用値

job `72946`、c396、12 組 interleaved A/B。入力は
`namelists/perf_p7_gemm_fused.conf`。µs/stage は 19 step × 3 RK = 57 で割った。

| 構成 | device s / 57 stage | µs/stage | Main ms/step | 対基準 |
|---|---:|---:|---:|---:|
| 旧準拠構成（y 64x32 s3、z 64x32 s4） | 0.144376 | 2532.9 | 7.731 | — |
| 最終（y 32x64 s3、z 16x32 s3） | **0.111735** | **1960.3** | **6.042** | **−22.61% device / −21.85% Main** |

同じ最終バイナリの `GEMM_CUTE` は device **1883.6 µs/stage**、Main
**5.815 ms/step**（6 run 平均）。したがって p=7 では z assembly 融合 package
が **+4.1%**で、融合自体は負ける。これは経路を置換する理由ではなく、
`CUTE` 対 `GEMM_FUSED` 比較が意図どおり fusion package の価格を測れている。

理論仕事量 1.405 GFLOP/stage、unique 1577 MB、経路 244 B/node = 4093.6 MB
から、最終 FUSED は **0.717 TFLOP/s**（40.1 の 1.8%）、unique
**0.805 TB/s**（7.9 の 10.2%）、path **2.09 TB/s**（26.4%）。

### 10.2 tile / schedule アブレーション

採否は全て同一ノード内 interleaved A/B。job `72555`--`72572` は c163、
`72921`--`72943` は c164。

| knob | 結果 | 判定 / 機構 |
|---|---:|---|
| y 64x64 s4 → 64x32 s4 | CUTE device 約 186.3 → 131.9 ms/57 | 採用候補。8x8 問題の predicated tile を縮小 |
| y 64x32 s4 → 32x64 s4 | 131.9 → 129.8 ms | 採用 |
| y 32x64 s4 → s3 | 129.8 → 120.7 ms | 採用。s2 は CUTLASS specialization ambiguity で不成立 |
| x wide → strided-batched 8x64 | GF 約 141.6 → 141.9 ms | 不採用。event の volume 部分値は overlap のため採否に使わない |
| x/y overlap を外す | 約 141.6 → 142.7 ms | overlap 採用（約 0.8%） |
| z 64x32 s4 → 32x64 s4 | 約 141.5 → 119.5 ms | 採用候補 |
| z 32x64 s4 → 32x32 s4 | 122.7 → 119.5 ms | 採用候補 |
| z 32x32 s4 → s3 | 122.7 → 121.1 ms | s3 採用 |
| z 32x32 s4 → 16x32 s4 | 122.7 → 113.8 ms | orientation/小 tile 採用 |
| z 16x32 s4 → s3 | 113.8 → **111.7 ms** | 最終採用 |
| z 64x16 s4 | 約 157.6 ms | 不採用。N を狭め過ぎる |
| z 32x16 s4 | 約 133.6 ms | 不採用 |
| z 16x16 | compile-time divisibility assertion | 不成立。合法な 2-warp geometry を作れない |
| z 64x32 s4 → s3 | CUTE は約 −1.9%、GF は悪化 | 不採用。優先する production GF の壁時間で決定 |

y の ncu（job `72567`、c163、`--set full`）では generic 64x64 s4 →
32x64 s3 により、先頭 y chunk が 1.003 → 0.492 ms、実行命令 277.1M →
149.0M、dynamic shared 65,536 → 36,864 B/block。達成占有率は
18.07% → 18.09% と不変で、DRAM も 0.44% → 0.88%。L1/TEX 律速のまま、
predicated tile の命令と shared footprint を半減したことが速さの理由である。

z の ncu（job `72945`、c396、同一 job の `--set full`）では 64x32 s4 →
16x32 s3 により、先頭 fused-z が 1.503 → **0.531 ms**、命令 506.1M →
176.8M、dynamic shared 49,152 → 18,432 B/block、register 254 → 128、
達成占有率 12.25% → **24.06%**。L1/TEX は 54.2% → 53.4% と同じだが、
DRAM 利用率は短時間化に伴い 8.7% → 24.5%。過大な M tile の predicated
仕事を除き、resident CTA を倍増したのが z 改善の機構である。

nsys job `72947`（c397、60 stage）の最終内訳は y CUTLASS 65.863 ms
（**1097.7 µs/stage**、5 chunk launch/stage）、fused z 21.598 ms
（**360.0 µs/stage**）、x cuBLAS 10.985 ms（183.1 µs/stage）、elembnd
10.139 ms（169.0 µs/stage）、volume flux 7.552 ms（125.9 µs/stage）。
旧 64x32 z の nsys 52.729 ms / 60 = 878.8 µs/stage に対し、最終 z は
**−59.0%**。残る最大項は y である。

### 10.3 数値・再現条件

job `72946`（c396）で `SCALE_DG_VARYING_COEFF=1` とし、p=7、Ne=2³ の
`CUDAFORTRAN_SPLIT` と最終 `CUDAFORTRAN_GEMM_FUSED` の owned `dqdt` 全点を比較。
最大絶対差 **1.77636e-15**、相対差 **2.86031e-16**。

プロファイラは CUDA graph off の凍結バイナリを用いた。再現コマンドの要点:

```bash
export DEBUGINFOD_URLS=
timeout 180 nsys profile --stats=true --resolve-symbols=false \
  -o output/nsys_p7gf_zfinal \
  ./scale-dg_extraction.p7gf_compliant_z1632s3 namelists/perf_p7_gemm_fused.conf

timeout 180 ncu --set full --csv --kernel-name-base demangled \
  -k 'regex:.*GemmBatchedDqdtAssembly.*' -s 0 -c 1 \
  ./scale-dg_extraction.p7gf_compliant_z1632s3 /tmp/p7gf_ncu_${SLURM_JOB_ID}.conf
```

## 11. y の batch を 1 ローンチにした（2026-09-01、採用、−3.3%）

p=15 の [`p15_gap_study.md`](p15_gap_study.md) §25.3 で入れた次数非依存の変更。
在庫の device-level `GemmBatched` は batch を `% 65536` して `grid.z` に載せる
ため、`Nq*Ne` が 65535 を超える次数では 65535 件ずつのチャンク launch にして
いた。p=7 は `Ne=32³` で y batch が 262,144 なので **5 ローンチ**である。
`kernel::GemmBatched` は `batch_idx += gridDim.z` で歩くので、Params を手で
組んで `grid.z` を丸めれば 1 ローンチで足りる（`run_gemm_batched_nn_capped`、
`cuda_cutlass_gemm_fused.cu`）。tile も warp も stage も変えていない。

Slurm job `74632`（c178）、§10 と同じ `namelists/perf_p7_gemm_fused.conf` を
12 回交互 A/B した中央値:

| 構成 | Main [ms/step] | device [ms/57 stage] | µs/stage | 対 §10 |
|---|---:|---:|---:|---:|
| §10 の最終形（5 launch） | 6.04200 | 111.682 | 1959.3 | — |
| **単一 launch** | **5.85210** | **108.030** | **1895.3** | **−3.27%** |

device のレンジは 111.535--111.772 対 107.855--108.142 で重ならない。基準側は
§10 の 1960.3 µs を再現している。最終形は **0.741 TFLOP/s**（40.1 の 1.8%）、
unique **0.832 TB/s**（10.5%）、path **2.16 TB/s**（27.3%）。

数値は `SCALE_DG_VARYING_COEFF=1`、`val_p7_split.conf` 対
`val_p7_gemm_fused.conf` で max abs **1.77636e-15**、相対 **2.86031e-16**
（§10.3 と同じ）。最速は `FUSED_TC` のまま。

## 12. y epilogue の repad（2026-09-01、採用、−4.1%）

[`p15_gap_study.md`](p15_gap_study.md) §26.2 の横展開。batched volume GEMM の
エピローグは accumulator の置き場を shared に取るが、在庫の設定では
半ワープの 16 レーンが同じバンク組に落ちる。z assembly が使っている
`RepadEpilogue<..., 8>` を batched launcher（`run_gemm_batched_nn_capped`）
にも入れ、行ストライドに 8 doubles 足した。mainloop・tile・stage は不変で、
`GEMM_CUTE` と `GEMM_FUSED` の両方に同じものが入る。

job `74636`（c178、12 回交互 A/B）:

| 構成 | Main [ms/step] | device [ms/57 stage] | µs/stage | 対 §11 |
|---|---:|---:|---:|---:|
| §11（単一 launch） | 5.85431 | 108.078 | 1896.1 | — |
| **repad 8** | **5.62624** | **103.634** | **1818.1** | **−4.11%** |

レンジは 108.036--108.239 対 103.452--103.841 で重ならない。p=15 では
同じ変更が −0.63%、p=63（y batch 4096）では差が無い。p=7 で効きが大きいのは
y の batch が 262,144 で、この kernel がステージに占める割合が高いためである。

最終形は **0.773 TFLOP/s**（40.1 の 1.9%）、unique **0.867 TB/s**（11.0%）、
path **2.25 TB/s**（28.5%）。数値は `val_p7_split.conf` 比で max abs
**1.77636e-15**、相対 **2.86031e-16**（§10.3 と同じ）。最速は `FUSED_TC` のまま。

## 13. `Nq>64` に閉じていた z epilogue の 4 つを p=7 で測る（2026-09-01、3 つ採用、−4.07%）

出所は [`p255_gap_study.md`](p255_gap_study.md) §10 と
[`p127_gap_study.md`](p127_gap_study.md) §12--13。あちらで
`GEMM_FUSED` の z assembly epilogue に入れた 4 つは、どれも
`cuda_cutlass_gemm_fused.cu` の `Nq > 64`（実際には `Nq >= 64`）の枝に閉じて
いた。p=255 §10 は「**次数が低いほど効きが大きい。epilogue の仕事は出力点数に
比例する一方 mma は `Nq^4` に比例する**」と書いているので、最も希釈の少ない
p=7（Nq=8）が最後の未測点である。

4 つを 1 つずつ独立に測れるようにするため、`EpilogueDqdtAssembly` の
`kWeighted` 1 個だったテンプレート引数を 3 個（`kWeighted` / `kAffine` /
`kPaired`）に割り、`GemmZWide`（16 B epilogue アクセス）を選ぶ `kWide` と
合わせて 4 ビットにした。既定値は `= kWeighted` なので `Nq > 64` の枝は
1 文字も変わらない。計測中は環境変数 `SCALE_DG_Z_EPILOGUE_OPTS` でビットを
選び、**1 本のバイナリに全変種を同居させて**同一ジョブ内で交互に測った
（採用後は p=7 の当たり値をコンパイル時に固定し、環境変数は削除した）。

| bit | 内容 | p=255 での値 |
|---:|---|---|
| 0 | 添字クランプをタイル原点へ集約（`kAffine`） | −0.85% |
| 1 | lift の 6 ロードを 3 本の 16 B ロードに（`kPaired`、面平面の (1,3)(2,4)(5,6) インターリーブ + `dg_lift_zpair`）。lift の積和をアキュムレータの shared 往復の後ろに回すのも同じビット | −0.75% |
| 2 | 16 B epilogue アクセス（`GemmZWide` = `EpilogueOp2`） | −1.6%（p=127 で 7.2 µs） |
| 3 | `Escale_x`/`Escale_y` と `deriv_x` を y epilogue へ前送りし、z が読む体積テンソルを 5 → 2 本に（`kWeighted`） | −1.0% |

### 13.1 採用値（Slurm job `75867` / `75869`、c363 / c182、12 回交互 A/B）

入力は `namelists/perf_p7_gemm_fused.conf`（`Ne=32³`、`nstep=20`、graph off）、
実行ファイルは §12 の親（`4189e3d` の作業ツリー）に本節の変更だけを載せた凍結
コピー。µs/stage は `CUDA device GEMM fused` を 19 step × 3 RK = 57 で割った値、
12 回の中央値。

| bits | 変種 | device [s/57 stage] 中央値 | min--max | µs/stage | 対 base |
|---:|---|---:|---|---:|---:|
| 0 | base（§12 の最終形） | 0.103395 | 0.103173--0.103548 | 1813.9 | — |
| 1 | クランプのみ | 0.106318 | 0.106167--0.106483 | 1865.2 | **+2.84%** |
| 2 | 16 B lift のみ | 0.100917 | 0.100825--0.101126 | 1770.5 | **−2.40%** |
| 4 | 16 B epilogue のみ | 0.100376 | 0.100227--0.100559 | 1761.0 | **−2.92%** |
| 3 | クランプ + 16 B lift | 0.100882 | 0.100590--0.100938 | 1769.9 | −2.42% |
| 5 | クランプ + 16 B epilogue | 0.100755 | 0.100611--0.100877 | 1767.6 | −2.54% |
| 6 | 16 B lift + 16 B epilogue | 0.100408 | 0.100314--0.100656 | 1761.5 | −2.89% |
| **7** | **3 つとも（採用）** | **0.099173** | 0.099045--0.099367 | **1739.9** | **−4.07%** |
| 8 | `Escale` 前送り（y fold） | 0.14425（login 3-run） | — | 2531 | **+39.4%** |

job `75867` は bits 0/2/4/6/7 を、job `75869` は bits 0/1/3/5/7 を測っている。
両ジョブの base（0.103395 / 0.103382）と bit 7（0.099340 / 0.099173）は
互いに 0.2% 以内で、bit 7 のレンジは base のレンジと重ならない。
採用値は **1813.7 → 1739.9 µs/stage（−4.07%）**、Main は 5.612 → 5.394
ms/step（−3.88%）。

**単独では負ける bit 0 が、bit 2|4 の上では −1.06% になる**（bit 6 の 1761.5 →
bit 7 の 1739.9）。p=63 / p=15 で「クランプ集約はビット一致のまま ptxas の
コード生成を悪くする」と測った現象そのものだが、符号が組み合わせで反転する
ことは今回が初めての観測である。**したがって bit 0 は「p=7 でも単独では
+2.84% の負け」と「3 つ揃えれば勝ち」の両方が正しい。**

最終形は理論仕事量 1.405 GFLOP/stage、unique 1577 MB、経路 4093.6 MB から
**0.808 TFLOP/s**（40.1 の 2.0%）、unique **0.906 TB/s**（11.5%）、path
**2.353 TB/s**（29.8%）。最速は `FUSED_TC`（274.9 µs/stage）のままで、
主比は 6.33 倍である。

### 13.2 機構（ncu job `75870`、c187、`--set full`、同一ジョブ内で 4 変種）

`GemmBatchedDqdtAssembly` の先頭ローンチ 2 本。

| | p=7 base | p=7 bits 7 | p=31 base | p=31 bits 7 |
|---|---:|---:|---:|---:|
| duration [µs] | 529.1 | **426.7**（−19.4%） | 507.5 | **1100.3**（+116.8%） |
| `smsp__inst_executed.sum` | 176.8 M | **143.7 M**（−18.8%） | 138.0 M | 129.9 M（−5.9%） |
| register / thread | 128 | 164 | 254 | 255 |
| local memory spilling requests | 0 | **0** | 0 | **16.3 M（overhead 100%）** |
| 達成占有率 [%] | 24.04 | 17.86 | 12.20 | 12.30 |
| Compute (SM) throughput [%] | 50.3 | 50.9 | 41.3 | 17.8 |
| warp cycles / issued inst | 7.55 | 5.53 | 4.68 | **10.89** |
| dynamic shared / block [B] | 18,432 | 18,432 | 49,152 | 49,152 |

**p=7 では時間が命令数にそのまま比例する**（−19.4% 対 −18.8%）。これは
p=255 §10.2 が z epilogue について測った「命令発行律速」と同じ関係で、
p=7 の z タイルが 16×32・shared 18 KB・レジスタ 128 と**予算に余裕がある**
（レジスタが 128 → 164 に増えても block limit は 8 → 6 で、スピルは 0）ため、
命令を減らす 3 つがそのまま時間になる。

**p=31 で同じ 3 つが崩壊する理由は 1 行で書ける: スピルする。**
z タイルは 64×32・shared 49 KB・レジスタ **254** で既に上限に張り付いており、
`kPaired` が要求する `lift_a` / `lift_b`（アキュムレータの shared 往復を
またいで生かす `double2` の配列）を置く場所が無い。ptxas はローカルメモリへ
落とし、16.3 M 回の spilling request（overhead 100%）で 1 命令あたりの待ちが
4.68 → 10.89 サイクルになる。**命令数は 5.9% 減っているのに時間は 2.17 倍**で、
これは「命令数が律速」という p=7 / p=255 の関係が p=31 では成り立って
いないことの直接の証拠である。同じ理由で Nq=16 / Nq=64 でも負ける
（[`p15_gap_study.md`](p15_gap_study.md) §25.8 の +2.3%、
[`p255_gap_study.md`](p255_gap_study.md) §10.4 の +2.7% / +0.8%）。

**bit 3（`Escale` 前送り）が p=7 で +39.4% と大敗する理由は別で、経路の
配分にある。** `Nq <= 64` では x が cuBLAS で epilogue を持てないので、
前送りの受け皿は y だけになり、y は `Escale_y` / `Escale_x` / `deriv_x` の
**3 本の体積テンソルを新たに読む**。p=7 の内訳（§10.2 の nsys、y 1097.7 / z 360.0 / x 183.1 / elembnd 169.0 /
volume flux 125.9 µs）では y がステージの約 57% を占める最大カーネルで、
z は約 19% しかない。z が 5 → 2 本で
得るものより、最大カーネルの y が 3 本増やす代金の方がはるかに大きい。
p=15 の +3.8%（§25.8: z の天井 −47 µs に対し y が +72 µs）と p=31 の +2.3%
と同符号で、**次数が下がるほど y の比重が上がるので赤字も大きくなる。**
これは「epilogue の仕事は出力点数に比例するので低次数ほど効く」という
p=255 §10 の一般則が、**同じ epilogue でも「どのカーネルに置くか」の配分に
ついては逆向きに働く**ことを示している。

### 13.3 p=31（Nq=32）

同じ 4 ビットを p=31 の z assembly（`MmaSet_884` の 64×32 タイル）に当てた。
入力は `namelists/perf_p31_gemm_fused.conf`。bit 3 は y に `GemmY32` の
重み付き双子が要る（= `GEMM_CUTE` と mainloop を共有しなくなる）ので実装せず、
§23 の +2.3% を引き継ぐ。

| bits | µs/stage | 対 base | 測定 |
|---:|---:|---:|---|
| 0（base） | 7400.1 | — | job `75869`、12 回交互（0.421081--0.422108） |
| 2 | 7469.4 | **+0.94%** | 同上（0.425043--0.426369、レンジ非重複） |
| 1 | 7617 | +2.8% | login 3-run |
| 4 | 8438 | +13.9% | login 3-run |
| 3 | 13,950 | +88.4% | login 3-run |
| 6 | 12,528 | +69.2% | login 3-run |
| 7 | 12,520 | +69.1% | login 3-run |

**採用ゼロ。** +13.9% 以上のものは login 3-run のレンジ（0.1% 未満）から
桁違いに離れているので占有 GPU での再測は要らない。機構は §13.2 の
スピルで、これで **Nq = 8 だけが 3 つを受け取れる次数**であることが
Nq = 8 / 16 / 32 / 64 / ≥128 の全点で測り終わった。

### 13.4 経路役割

- volume GEMM の mainloop は 1 バイトも触っていない。変えたのは z assembly
  の epilogue（`GEMM_FUSED` 固有の融合パッケージ）と、`GemmZWide` という
  **epilogue だけが違う同一 tile / warp / stage / swizzle** の型である。
- `Escale` 前送り（bit 3）は不採用なので、`GEMM_CUTE` の x/y epilogue は
  **非加重のまま**であり、z は `dqdt` に実体化されて
  `separable_lift_assembly` が最終重みと lift を行う定義を保っている。
- `elembnd_flux_kernel` の面平面インターリーブ（`pair_nq2`）は
  `fuse_epilogue` のときだけ有効で、`GEMM_CUTE` は従来のレイアウトを読む。
- `Nq<=64` の cuBLAS-x 分岐は両経路で共有のまま、追加のライブラリ分岐は無い。
- 同一バイナリの `GEMM_CUTE` は login 3-run で device 0.09827--0.09837 s/57
  = 1724--1726 µs/stage、改修前 0.098472 と一致し、点変化係数の owned `dqdt`
  は `CUDAFORTRAN_SPLIT` と**最大絶対差 0**。mainloop が動いていないことの
  確認になっている。

### 13.5 数値検証

`SCALE_DG_VARYING_COEFF=1`、`namelists/val_p7_split.conf`（`CUDAFORTRAN_SPLIT`）
を参照に owned `dqdt(:,1:Ne)` 全点（4096 点）を比較。測った全組合せで
同じ値である。

| 対象 | 最大絶対差 | 相対差 |
|---|---:|---:|
| p=7 `GEMM_FUSED` bits 0,1,2,3,4,8,9,10,11,15 | 1.77636e-15 | 2.86031e-16 |
| p=7 `GEMM_FUSED` 採用形（bits 7） | 1.77636e-15 | 2.86031e-16 |
| p=7 `GEMM_CUTE` | **0** | — |
| p=31 `GEMM_FUSED` bits 0--7 と採用形 | 1.77636e-15 | 4.19244e-16 |

§10.3 以来の値と同じで、`Escale` の掛け場所は変えていないので FMA の縮約も
変わっていない。C の interface は増減していないので `mod_cuda_dg_kernels_stub.f90`
の同期は不要である。
