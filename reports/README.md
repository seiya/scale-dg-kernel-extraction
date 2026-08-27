# Performance reports

GPU 実装と最適化の記録。すべて RIKYU の NVIDIA GB200 1 GPU 上での測定で、
ビルドは `make CUDA=1 GPUFLAGS=-gpu=cc100`（nvcc は `-arch=sm_100`）。

| ファイル | 内容 |
|---|---|
| [`overall_summary_report.md`](overall_summary_report.md) | 全実装パスの横断まとめ。時間内訳、ncu 効率分析、理論仕事量に対する達成率、不採用にした最適化とその理由。**最初に読むならこれ。** |
| [`execution_times.md`](execution_times.md) | `nstep=1000` の同一条件でのパス別実行時間 |
| [`gpu_optimization_session_report.md`](gpu_optimization_session_report.md) | OpenACC → CUDA Fortran → Tensor Core / GEMM に至る実装の変遷と、途中で踏んだ誤り（代表スカラー特殊化）の記録 |
| [`p255_gemm_fusion_session_report.md`](p255_gemm_fusion_session_report.md) | p=255 の volume GEMM と z-epilogue 融合の詳細実験 |
| [`tc_paper_survey_2407.09621.md`](tc_paper_survey_2407.09621.md) | arXiv:2407.09621 の取り込み調査と、p=7 Tensor Core カーネルの shared memory レイアウト刷新 |
| [`h100_report.md`](h100_report.md) | H100（TSUBAME 4）で同じコードを走らせた記録。経路横断の GB200 比、FP64 Tensor Core ピークが 2 倍あることの帰結、H100 では `CutlassMmaShape = "16x8x4"` を選ぶこと |
| [`sm90_mma_shape_survey.md`](sm90_mma_shape_survey.md) | CUTLASS volume GEMM の MMA 命令形状（8x8x4 / 16x8x4 / 16x8x8 / 16x8x16）を namelist で選べるようにして実測した記録。GB200 では ptxas が SM90 の f64 MMA を `DMMA.8x8x4` に展開するため得るものが無く、H100 では 16x8x4 が最速（cuBLAS が選ぶ 16x8x8 ではない）。kK>4 を CUTLASS 2.x で正しく動かすための warp tile iterator も含む |
| [`tma_survey.md`](tma_survey.md) | TMA の適用可能性を候補ごとに実測した記録。採用ゼロだが、FP64 での受理条件・帯域・L1 挙動と、2 候補それぞれの構造的な不採用理由 |
| [`cublas_emulation_survey.md`](cublas_emulation_survey.md) | cuBLAS FP64 fixed-point emulation の見かけのストール調査。EAGER 強制、永続 8 GiB workspace、p=7/p=255 の速度と数値検証 |
| [`ozaki2_survey_2504.08009.md`](ozaki2_survey_2504.08009.md) | arXiv:2504.08009v3（Ozaki Scheme II、INT8 Tensor Core による FP64 GEMM エミュレーション）の適用調査。不採用だが、成立条件が `p ≳ 500-650` であること、およびハードウェア条件が 3.82 FLOP/byte であることを実測から確定した |
| [`p15_gap_study.md`](p15_gap_study.md) | p=7 と p=255 の間を同一 DOF で埋める最初の点 p=15 (Nq=16)。CUDA core 版と Tensor Core 版の融合カーネル、shared 戦略、Nq=16 では融合したまま占有率 50% を超えられないという構造的な壁 |

## 現時点の結論

- **Ozaki Scheme II（arXiv:2504.08009v3、2026-08-27）**: INT8 Tensor Core による
  FP64 GEMM エミュレーションは **不採用**。GB200 の INT8 天井は 4724 TOP/s
  （FP64 の 118 倍）で論文の前提は満たすが、volume GEMM の `K = Nq` が浅く、
  K=256 では INT8 GEMM が INT32 出力の書き出し帯域で律速して天井の 7.6% しか
  出ない。s=14 のエミュレーションは**演算だけ**で native DGEMM の 95-103% を
  消費し、CRT 再構成を足すと 1.9 倍遅い。融合実装も s 組の INT32
  アキュムレータ（917 KB/CTA）がレジスタ 256 KB に入らず不可能。
  実測レートからの外挿では成立条件は **p ≳ 500-650**（確実に勝つのは p ≳ 1000）で、
  p=255 は交点の半分の Nq しかない。x/y/z をまとめても 1.83x → 1.63x で桁は変わらない
  （`Escale` の点ごと重み付けにより 3 方向は INT32 アキュムレータを共有できない）。
  ハードウェア側の条件は `P_fp64 / B < 2*Nq/(9s+8)` = **3.82 FLOP/byte**（p=255, s=14）で、
  GB200 の 5.08 は **1.33 倍足りないだけ**。鍵は INT8 の速さではなく FP64 に対する帯域である。
  詳細は `ozaki2_survey_2504.08009.md`。

- **cuBLAS FP64 emulation（2026-08-27）**: `CublasEmulation=.true.` は比較実験の
  ため `EAGER` を明示的に強制する。p=7 では native FP64 の **約131倍遅い**ため、
  計測は `nstep=1--10` に制限する。8 GiB workspace は初期化時に一度だけ確保して
  再利用するが、cuBLAS 13.2.1 の内部 `cudaMallocAsync` は残り、p=7 の不利は
  解消しない。詳細は `cublas_emulation_survey.md`。

- **p=7, `Ne=32^3`**: `CUDAFORTRAN_FUSED_TC` が最速（commit `e22dda1` 以降）。
  現時点の device 時間は 0.806 秒 / Main 1.066 秒（下の ±x 面の項目）。
  それ以前は `CUDAFORTRAN_FUSED` が最速だった。
  さらに occupancy を 100% に上げる作業（`tc_paper_survey_2407.09621.md` §7）で
  device 時間 1.076 → 0.851 秒。同じ知見を CUDA core 版にも適用すると
  そちらも 1.153 → 0.986 秒になり（§8）、両者を 100% occupancy で揃えた
  TC 版の優位は 1.16× である。
- **p=255, `Ne=1`**: `CUDAFORTRAN_GEMM_FUSED` が最速。手書きの Tensor Core 経路は
  CUTLASS / cuBLAS の multistage mainloop に大きく負ける。
- 同じ体積 DOF 数でも、p=7 と p=255 で最適戦略は逆転する。
- **p=15, `Ne=16^3`（2026-08-27）**: 同一 DOF の 3 点目。同一 DOF を立方一様メッシュで
  保つ条件 `NeX*Nq = 256` から、間を埋められる次数は **p = 15, 31, 63, 127 の 2 冪だけ**で、
  この条件は FP64 mma のタイル条件より強い。p=15 は**要素まるごとが shared に載る最後の点**
  （1 要素の `q` が 32 KB、Nq=32 では 256 KB で 227 KB/SM を超える）。
  `CUDAFORTRAN_FUSED` に `tendency_fused_p15_kernel` を追加した。Nq³ のスクラッチバッファ
  1 本を x/y/z と面流束で使い回し `q` をレジスタに置くことで、global ロードを理論最小に
  保ったまま shared 35584 B の静的枠に収めている。レーンが `i` 方向に並ぶので
  **swizzle なしで 3 方向とも conflict free**。device 時間は **p=7 の 1.31 倍で 2 倍の
  体積演算**（433.9 対 331.0 µs/stage）、p=15 における従来最速 `OPENACC_SPLIT` の 2.1 倍速い。
  重要なのは**律速の性格が変わったこと**で、p=7 がメモリ系 95.6% で飽和しているのに対し
  p=15 は**どこも飽和していない**（メモリ 74%、SM 36%、DRAM 40%）。
  `launch_bounds(1024,2)` は 32 レジスタ spill ゼロを達成しながら **+6.8%**、
  carveout 追加で **+8.2%** と、どちらも不採用。
  **Tensor Core 版も実装した**（`tendency_fused_p15_tc_kernel`）。m8n8k4 のタイルが
  8x8 なので 16x16 プレーンは出力タイル 4 枚・k ステップ 4 回になるが、
  **フラグメント配置 1 つが x/y/z 3 方向すべてに使え、swizzle も 1 本で x と y を賄える**。
  shared 戦略は CUDA core 版のものをそのまま流用でき、事前調査が TC 移植の最大の障壁と
  見ていた「3 本の流束パネルで 96 KB」は問題にならなかった。
  結果は device **26.04 → 22.22 ms（1.17 倍）**で、ncu では狙いどおり
  **Compute (SM) が 36.0% → 19.8% とほぼ半減**、shared ロード命令も
  **31.2 M → 7.34 M（4.25 分の 1）**になっている。
  ただし**帯域は使い切らず**（DRAM 47%）、占有率も 48.9% で動かない。
  **CUDA core に対する TC の優位は Nq とともに伸びない**（p=7 で 1.21 倍、p=15 で 1.17 倍）。
  両版が同じ占有率で並ぶのは構造的で、z 収縮が要素全体を要求する以上
  1 要素 = 1 ブロック、4096 ノード / 最大 1024 スレッド = 4 ノード/スレッドが下限、
  その状態量は 32 レジスタに収まらない、という連鎖による。
- **占有率は律速ではなかった（2026-08-27、当初の結論を訂正）**: 上記までこの調査は
  占有率 49% を犯人としていたが、それは「どのユニットも飽和していない」という
  状態証拠だけによるもので、**占有率が実際に上がった版を一度も測っていなかった**。
  レジスタ 32 本 + carveout 50% で **占有率 96.7% を達成した版は、49% の baseline より
  8.0% 遅い**（26.03 → 28.14 秒相当、ncu 679.4 → 733.8 µs）。レジスタ絞りを揃えて
  比べた占有率上昇そのものの正味価値は **−1.2%**。ncu では warp を倍にすると
  **memory pipe が 74% → 89% に上がるだけでカーネルは短くならない**、
  すなわち増えた warp はレイテンシを隠さず同じ L1/TEX 経路を混ませていた
  （carveout 100% で L1 を潰すと +52% になるのも同じ話の裏側）。
  **実際の律速は global ロードの L1/TEX トランザクション**で、支配的な stall は
  long scoreboard 20〜22。最大の塊は面フェーズ（カーネルの 35%）の、
  特に隣接要素を読むため手の出せない P 側 gather である。
  **面 gather の追加調査（2026-08-27）**: p=7 で −5.1% だった ±x 面 M 側の
  shared ステージングは、Nq=16 ではアブレーション実測で**上限 2.0%** にしかならず
  （面点率が半減したため）、レジスタにも shared にも置き場が無いので**不採用**。
  ±x 面は P 側の方が大きい（7.5%）が隣接要素なので原理的に手が出ず、
  面フェーズ全体（カーネルの 35%）も単独カーネルに出すと逆に高くつく。
  代わりに**shared load の 4-way コンフリクトが 2 か所**見つかった（z 微分の往復と
  面 5/6 の lift 読み出し。いずれも書き手と読み手で添字の形が違うことによる）。
  死にビットに畳む swizzle 2 本で **21.78 ms（−2.1%）**、
  コンフリクト **−93.7%**、stall short_scoreboard −31%。
  レジスタも shared も増やさず**ビット一致**。Nq=8 では完全 conflict free 化が
  遅かった（survey §10.3）が、あちらは 2-way → 1-way、こちらは 4-way → 1-way で桁が違う。
  最終形は **363.0 µs/stage、5.72 TFLOP/s（14.3%）、3.75 TB/s（47%）**。
  詳細は `p15_gap_study.md`。
- **p=7 TC の整数・アドレス演算削減（2026-08-25）**: `tc_paper_survey` §10 が
  次の標的に挙げた整数演算は、単独で削っても end-to-end では効かない（§11）。
  同じ調査で見つかった効く要因は global アクセスのキャッシュライン数で、
  x/y 導関数をレジスタに保持し mma 出力を転置して epilogue を coalesce
  させた版が device 0.8518 → 0.8488 秒。残る律速は global load レイテンシ。
- **1 step の CUDA Graph 化（2026-08-25）**: SSP-RK3 の 1 step を 1 回だけ
  捕捉して以降は再生するようにした（namelist `UseCudaGraph`）。カーネル・順序・
  データは不変で、消えるのは launch turnaround だけ。Main は p=7 `FUSED_TC` で
  1.2038 → 1.1716 秒、p=7 `FUSED` で 1.3441 → 1.3104 秒、p=255 `GEMM_FUSED` で
  3.9730 → 3.8545 秒、p=255 `GEMM_CUTE` で 4.2646 → 4.1328 秒。
  **再生中は Fortran のラッパを通らないので、このモードでは tendency の
  CUDA event 時間が採れない**（`execution_times.md` 追記 7、
  `overall_summary_report.md` §8.3）。
- **p=7 TC の face gather 前倒し（2026-08-25）**: `tc_paper_survey` §11.6 が
  次の標的に挙げた「`VMapM`/`VMapP` の先行ロードで 2 段依存を volume の
  ロードと重ねる」案は**不採用**（同 §12）。index だけ前倒しした版は
  device 0.850 → 0.868 秒（+2.1%）、セクションごと入れ替えた版は ±0。
  カーネルは 8 ブロック/SM を保つためレジスタ 32 本ちょうどで、先行ロードを
  保持する余地が構造的に無い。加えて occupancy 96.9% では
  スレッド内 MLP を増やす意味がなく、残る余地は L1 トランザクション数の
  削減側だけ（その方向は §9 / §10 で不採用済み）。
- **p=255 の lift から中間配列を消した（2026-08-25）**: lift は
  `Lift1D` と 6 面から作る 3 つの rank-2 項の和で、`K=2` の cuBLAS GEMM 3 本が
  `lift_out`（134 MB）に `beta=1` で累積していた（670 MB の往復）。
  まず 1 本の `separable_lift_kernel` にまとめ（`GEMM_FUSED` 3.971 →
  3.635 秒）、次に z-epilogue 内で 6 面から直接評価して `lift_out` 自体を
  消した（→ **3.463** 秒、通算 **−12.8%**）。旧実装と**ビット一致**。
  `GEMM_CUTE` 4.279 → 3.954、`GEMM` 4.241 → 3.946 は前段のみの効果。
  **素直に epilogue へ移すと −0.6% にしかならない**（出力 1 点ごとの
  `p % Nq` / `p / Nq` が SM 律速の z GEMM に効く）ことが重要な知見で、
  column 不変量を `kIterations` ループ外に括り出して初めて −4.7% になる。
  詳細は `overall_summary_report.md` §8.4 / §8.5 と
  `execution_times.md` 追記 8 / 9。
- **`volume_flux_kernel` のロードをまとめた（2026-08-25）**: GEMM 系に残る唯一の
  独立カーネル（150 µs、ncu DRAM 66%）を ncu で測ると、DRAM read は理論値の
  **1.000 倍**でセクタ効率も 100%、つまりトラフィックには一切無駄が無い一方、
  **どのユニットも飽和していなかった**（DRAM 65.7 / L1 52.5 / SM 33.7%）。実体は
  帯域律速ではなくレイテンシ律速で、`q(idx)*u(idx)` を 3 行並べた書き方のせいで
  1 warp あたりの global load が 4 命令ではなく 6 命令になっていた。4 本の
  ロードをストアより前にまとめると **150.6 → 125.9 µs（−16.4%、DRAM 83.4%、
  7.09 TB/s = ピークの 90%）**。Main は p=255 `GEMM_FUSED` 3.4469 → **3.3702** 秒、
  p=7 `CUDAFORTRAN_SPLIT` 2.7172 → **2.6440** 秒で、旧実装と**ビット一致**。
  `q` だけをレジスタに退避した版は効かない。詳細は
  `overall_summary_report.md` §8.6 と `execution_times.md` 追記 10。
- **現行の最速パスの達成効率（2026-08-25、`78fbbf8`）**: p=7 `FUSED_TC` は
  tendency 277.5 µs で **5.02 TFLOP/s（ピークの 12.5%）/ 5.75 TB/s（72.7%）**、
  p=255 `GEMM_FUSED` は 966.9 µs で **27.01 TFLOP/s（67.3%）/ 2.62 TB/s（33.1%）**。
  **同じ 16.78 M 自由度でも p=7 は帯域・L1 律速、p=255 は演算律速**と性格が
  正反対である。詳細は `overall_summary_report.md` §7.1。
- **全カーネルのロード監査と SSP-RK 更新の 1 次元化（2026-08-25）**:
  1 warp あたりの global load 命令数をソース上の相異なるロード数と突き合わせて
  全カーネルを監査したところ、**CUDA カーネルはすべて最小**で、外れたのは
  時間発展ループの OpenACC RK 更新 2 本だけだった。ただし原因の `rk_a(stage)`
  を直しても −0.2% で、**本当の律速は `collapse(2)` が要求する実行時値 `Np`
  での整数除算**だった。連続領域なので 1 次元ループに直すと、カーネルは
  SM 62.7% / DRAM 41.7% の演算律速から SM 35.3% / DRAM 74.2% の帯域律速に
  変わり、321 → 185 µs/step。tendency に触らないので**全パスが得をする**:
  p=7 `FUSED_TC` 1.208 → **1.131** 秒（graph on 1.171 → **1.083**）、
  p=255 `GEMM_FUSED` 3.329 → **3.232**（graph on 3.289 → **3.192**）。
  ビット一致。詳細は `overall_summary_report.md` §8.8 と
  `execution_times.md` 追記 12。
- **境界流束を GEMM の裏に隠した（2026-08-25）**: x/y GEMM は SM 88–89% で
  回りながら DRAM を 6–10% しか使わないので、その裏に帯域律速の
  `elembnd_flux_kernel`（19.6 µs）を 2 本目のストリームで流し込んだ。nsys で
  x GEMM の区間に完全に収まることを確認。Main は p=255 `GEMM_FUSED`（graph off）
  3.3723 → **3.3293** 秒（−1.3%）、`GEMM_CUTE` 3.8820 → 3.8368。ビット一致。
  **volume flux を方向で割って隠す案は不採用**で、理由は DRAM ではなく
  レジスタ（x GEMM は SM あたり 11,264 本しか残さない）。**CUDA Graph の
  replay では損になる**ので graph モードでは 1 本に戻している。詳細は
  `overall_summary_report.md` §8.7 と `execution_times.md` 追記 11。
- **p=7 の ±x 面 M 側 gather を shared 経由にした（2026-08-26）**: 全カーネルを
  「何に律速され、上限に対してどこまで出ているか」で棚卸ししたところ
  （`overall_summary_report.md` §13.1）、余地があるのは p=7 の tendency
  （L1/TEX 91.8% 張り付き）だけだった。`-lineinfo` 付きの ncu Source ページで
  超過セクタ 2477 万の帰属を命令単位に取ると、**6 面のうち ±x 面（面 2, 4）の
  gather 8 本に全部乗っていた**。`Fmask` がその 2 面に `i0 + 8j + 64k` を
  与えるので warp が 8 doubles 飛びになり、32 B セクタの 8 B しか使わない。
  M 側は同じ要素の値で volume ロードが既にレジスタに持っているので、
  4.6 KB の shared にステージングして読む。global load sectors
  **95.9 M → 78.6 M（−18%）**、long scoreboard 35.1 → 24.0、device
  **0.8497 → 0.8060 秒（−5.1%）**、Main 1.1092 → **1.0656**（graph on は
  1.0738 → **1.0367**）。`63a4234` と**ビット一致**。
  ここで一番高くついた誤りは `cudaSharedmemCarveoutMaxShared` を反射的に
  足したことで、占有率は 1 ブロックも増えないまま L1 データキャッシュが削られ
  **+31%** になった。詳細は `tc_paper_survey_2407.09621.md` §13。
- **CUTLASS volume GEMM の MMA 命令形状を選べるようにした（2026-08-26）**:
  H100 の cuBLAS が同じ問題形状に `tensor16x8x8` を選ぶのに対し GB200 の
  cuBLAS は `d884`（8x8x4）を選ぶ、という食い違いを実測で解決した。
  namelist `CutlassMmaShape`（`8x8x4` / `16x8x4` / `16x8x8` / `16x8x16`）を
  追加し、`GEMM_CUTE` の 3 GEMM と `GEMM_FUSED` の融合 epilogue が従う。
  kK>4 の 2 形状は素の CUTLASS 2.x では**数値が壊れる**（64bit warp tile
  iterator が 4 深の K グループ前提）ので、kK=4 のイテレータを K 方向に
  積み直す `cutlass_f64_kdeep_mma.h` を足した。4 形状とも GB200 の
  `Ne=1` / `Ne=2` と H100 実機の DMMA の両方で参照と一致する。
  **GB200 では命令形状を変えても何も起きない**（4 形状が 0.1% 以内。ptxas が
  `mma.sync.m16n8k4/8/16.f64` を `DMMA.8x8x4` の 2 / 4 / 8 命令に展開するため。
  `sm_90` では 1 命令）。**H100 では逆に大きく効き、しかも効くのは cuBLAS が
  選ぶ 16x8x8 ではなく 16x8x4**（TSUBAME job `8502531`）。volume GEMM の
  device 時間は `GEMM_CUTE` で 2.907 → **2.225** 秒（−23%）、H100 の最速は
  `GEMM_FUSED` + `16x8x4` で Main 6.155 → **5.711** 秒（cuBLAS 5.832 も下回る）。
  カーネル単位では 16x8x8 の負けは **x GEMM 1 本に集中**（352.4 対 279.8 µs）。
  ncu によると原因は **レジスタ 255 本張り付きと 4.46 MB の spill**で、
  16x8x4 は 248 本で spill ゼロ（x の warp タイルは唯一 32x64x16）。
  8x8x4 は SM 86.9% の発行律速、16x8x4 はそれを実行命令数
  68.2 M → 56.2 M で解いている。volume GEMM 3 本の合計で
  cuBLAS との比は **1.68 → 1.27 倍**まで縮み、残りは mainloop の実装差
  （y GEMM は同一タイル・同一 stage でなお 1.23 倍）。
  16x8x4 なら **H100 の volume GEMM は GB200 より 3% 速い**（741.7 対 765.3
  µs/call）。**既定は GB200 の最速に合わせて `8x8x4` のまま**とし、H100 では
  入力に `CutlassMmaShape = "16x8x4"` を足す。詳細は
  `sm90_mma_shape_survey.md`、H100 全般は `h100_report.md`。
- **p=255 の RK 更新を z epilogue に融合するのは損（2026-08-26、不採用）**:
  RK 更新カーネルは DRAM 79–87% で、削り代は転送量にしかない。`dqdt` の
  往復 268 MB/stage を消すために z GEMM の epilogue で SSP-RK 更新まで
  済ませたが、**オペランドを 3 本足すと z GEMM が 338.8 → 481.1 µs（+42%）**に
  なり、RK カーネル 225.6 µs/step を消しても Main は 3.228 → 3.407 秒
  （+5.5%）。帯域ではなく 4 ブロック/SM で隠せないロードレイテンシが実体で、
  `tma_survey.md` §2 と同じ壁である。コードは差し戻した。p=7 側で同じことを
  する案は、P 側 gather が隣接要素の `q` を読むためブロック間レースになり、
  性能以前に成立しない。詳細は `overall_summary_report.md` §13.3。
- **z GEMM の shared store バンクコンフリクトを消した（2026-08-26）**: 下の
  「未着手」項目の決着。**帰属が違っていて**、犯人は `MmaMultistage` ではなく
  **epilogue のアキュムレータ smem ステージング**だった（mainloop は `cp.async`
  なので STS を 1 命令も出さない）。`DefaultEpilogueTensorOp` の
  `Padding = <0,4>` が行ストライドを 36 doubles = 72 word にしていて
  `72 mod 32 = 8`、連続 2 行のバンクが 8 本重なる 2-way。Padding を 8 にすると
  行ストライド 40 doubles で 2 行が 32 バンクを 1 回ずつ覆う。
  コンフリクト **1,165,739 → 136,379（−88%）**、store wavefronts
  2.21 M → 1.18 M（理想 1.05 M）。smem は mainloop と union なので**コストゼロ**、
  出力は**ビット一致**。ただし **Main は 3.2315 → 3.2313 秒で時間は動かない**。
  このカーネルは wait 36.6% / math pipe 20.4% が支配的で shared 経路では
  律速されていない。**ncu の「推定改善余地 30.5%」はその経路単独の上限であって
  カーネル時間の予測ではない**、が本件の一番の知見。副産物として、
  mainloop の LDGSTS 側にさらに大きなコンフリクト（z で 3.87 M）があるが、
  CUTLASS 側に調整の余地が無い。詳細は `overall_summary_report.md` §8.9 と
  `tma_survey.md` §2.3 / §2.4。
- **TMA は実測して採用ゼロ（2026-08-26、`tma_survey.md`）**: 前日の机上調査
  （`overall_summary_report.md` §12.9）の続きを実測した。まず前提として、
  現行の `-arch=native`(sm_100) では **CuTe の TMA は無効**で `sm_100a` が要る。
  FP64 の tensor map は作れて転送も正しいが、**DRAM が飽和している限り
  帯域の上積みは無い**（実形状 2 種で素のロードと 1.8% 以内）。TMA の正体は
  「速い転送」ではなく**「L1/TEX とレジスタと発行スロットを使わない転送」**で、
  同一バイトの対照実験で L1 wavefront −89%、global sector 1678 万 → **0**、
  shared store コンフリクト 128 万 → **0**、レジスタ 30 → 16 になる。
  そのうえで 2 候補とも不採用:
  **p=255 z GEMM の epilogue** は、TMA が隠せる long scoreboard stall が
  全体の **11.3%** しか無く（支配的なのは固定レイテンシ依存 36.6% と
  演算パイプ 20.4%）、occupancy は **shared memory** で 4 ブロック/SM に
  張り付いていてレジスタを返しても上がらない。増分時間の実体は
  5 本の追加オペランドの **701.7 MB/call** で、これは減らせない。
  **p=7 `FUSED_TC`** は L1/TEX 91.9%・long scoreboard 61.9% と理想的な標的
  だったが、**TMA が 8 B 要素で符号化できる 4 通りのレイアウト全部で
  x/y 収縮が 2-way バンクコンフリクト**になる。`sw_xy` が要求する
  「ノード bit4 → アドレス bit2」を TMA の swizzle 語彙が表現できない
  （行ビットは必ず最下位チャンクビットから当たるので bit4 は bit1 に落ち、
  n1 自身の像と衝突する）。§12.9 の「smem に余地が無い」は古く、実際は
  +8 KB まで 8 ブロック/SM を保てるが、落ちたのは予算ではなくレイアウトである。
- **z GEMM の shared store に 8.5-way バンクコンフリクトがある（2026-08-26、着手済み → 上の項目）**:
  上の調査の副産物。p=255 最速パスの単独最大カーネル（nsys 340.0 µs、
  tendency の 31.6%）で、shared store wavefronts の **52%** がコンフリクトで、
  ncu の推定改善余地は **28.8–30.5%**。融合版と CUTE 版で同値なので、
  自作 epilogue ではなく **CUTLASS 標準の `MmaMultistage`** 側にある。
  **TMA より期待値が高い次の標的**（`tma_survey.md` §2.2）。
- **tendency 以外（その 2）**: 時間発展ループの OpenACC 領域を `async` にし、
  CUDA / cuBLAS / CUTLASS のカーネルを同じストリームに載せて、カーネル間の
  GPU アイドルを 139 → 50 µs/step にした（2026-08-25）。Main は p=7 `FUSED_TC` で
  1.3099 → 1.207 秒、p=7 `FUSED` で 1.4412 → 1.344 秒、p=255 `GEMM_FUSED` で
  4.0890 → 3.960 秒。device 時間は不変。**この変更以降 `Cal_tend` は
  tendency の wall 時間ではない**（`execution_times.md` 追記 5、
  `overall_summary_report.md` §8.2）。
- **tendency 以外**: `q0 ← q` を SSP-RK stage 1 の更新カーネルに融合し、
  独立カーネルを削除した（2026-08-25）。非 tendency は約 422 → 320 µs/step、
  Main 時間は p=7 TC で 1.415 → 1.312 秒、p=255 `GEMM_FUSED` で
  4.189 → 4.097 秒。tendency 側の時間は不変。詳細は
  `execution_times.md` 追記 4 と `overall_summary_report.md` §8.1。

## 読むときの注意

各レポートは書かれた時点の commit と Slurm job を明記している。後の変更で
結論が変わった箇所には追記を入れてあるが、**表の数値そのものは当時の値のまま**
なので、commit 表記を確認すること。

最適化が守るべき数値契約はリポジトリ直下の `AGENTS.md` にある。
