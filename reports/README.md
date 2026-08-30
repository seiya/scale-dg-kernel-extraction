# Performance reports

GPU 実装と最適化の記録。すべて RIKYU の NVIDIA GB200 1 GPU 上での測定で、
ビルドは `make CUDA=1 GPUFLAGS=-gpu=cc100`（nvcc は `-arch=sm_100`）。
経路×次数の最新時間・アルゴリズム FLOP/s・DRAM スループットは下の
**「最新結果のまとめ」**。

| ファイル | 内容 |
|---|---|
| [`overall_summary_report.md`](overall_summary_report.md) | 全実装パスの横断まとめ。時間内訳、ncu 効率分析、理論仕事量に対する達成率、不採用にした最適化とその理由。**最初に読むならこれ。** |
| [`execution_times.md`](execution_times.md) | `nstep=1000` の同一条件でのパス別実行時間。**追記 16（2026-08-29）は p=7 の経路横断再測定**。追記 18 は p=7 CC 復活、追記 19 は p=15…255 の CC 測定、追記 20 は CC/DFMA 取り違えの独立確認、追記 21 以降は p=255 CC、**追記 30 は p=31 CC 融合の −27.6%**、**追記 31–32 は p=63 CC の 966.4 → 618.6 µs/stage** |
| [`gpu_optimization_session_report.md`](gpu_optimization_session_report.md) | OpenACC → CUDA Fortran → Tensor Core / GEMM に至る実装の変遷と、途中で踏んだ誤り（代表スカラー特殊化）の記録 |
| [`p255_gap_study.md`](p255_gap_study.md) | p=255 の `CUDAFORTRAN_FUSED_TC` を **1563.9 → 968.8 µs/stage（−38.1%、全段ビット一致）**にした記録。チャンクループの二重バッファ化と 1 ワープ 4×4 mma タイルは**組でしか効かない**、エピローグを b 外 / a 内に組み替えると LDG が 112 → 32 本になる、ストアのペア格納順で 2-way バンクコンフリクトが消える。**これで p=255 の最速は `GEMM_FUSED` から `FUSED_TC` に替わった。**命令数を 17〜34% 減らす変更が 2 度とも遅くなったことの記録も含む。**§10（同日）は逆に `GEMM_FUSED` 側を 1048.8 → 1021.2 µs/stage（−2.6%）にした**: z の assembly epilogue は**命令発行律速**（lift を消すと命令 −13.0% で時間 −12.8%、stall 内訳も占有率も不変）で、`Escale_x/y` を x/y GEMM の標準 epilogue に前送りし、添字クランプをタイル原点へ集約し、lift の 6 本のロードを 3 本の `double2` にした。手書き epilogue で標準 epilogue を置き換えると**それだけで +72 µs** かかる |
| [`p255_gemm_fusion_session_report.md`](p255_gemm_fusion_session_report.md) | p=255 の volume GEMM と z-epilogue 融合の詳細実験。末尾に **p=255 Tensor Core カーネルのタイル化**（2026-08-27）: 1 warp/block・オペランド全再読み込みで L1/TEX 99% に張り付いていた 3 本を、64×64 タイル・warp 2×4 register blocking の 1 本に統合して **2.86 倍**（4474.3 → 1566 µs/stage、ピーク比 14.6% → 41.6%）。それでも `GEMM_FUSED` には 1.57 倍負けるので **p=255 の最速は `GEMM_FUSED` のまま** |
| [`tc_paper_survey_2407.09621.md`](tc_paper_survey_2407.09621.md) | arXiv:2407.09621 の取り込み調査と、p=7 Tensor Core カーネルの shared memory レイアウト刷新。§14 は p=31 で効いた「D1D フラグメントのレジスタ常駐化」が p=7 では**効かない**ことの実測（32 レジスタの天井に余白が無く、shared ロード 2 本がスピル 2 本に置き換わるだけ、+0.63%） |
| [`h100_report.md`](h100_report.md) | H100（TSUBAME 4）で同じコードを走らせた記録。経路横断の GB200 比、FP64 Tensor Core ピークが 2 倍あることの帰結、H100 では `CutlassMmaShape = "16x8x4"` を選ぶこと |
| [`sm90_mma_shape_survey.md`](sm90_mma_shape_survey.md) | CUTLASS volume GEMM の MMA 命令形状（8x8x4 / 16x8x4 / 16x8x8 / 16x8x16）を namelist で選べるようにして実測した記録。GB200 では ptxas が SM90 の f64 MMA を `DMMA.8x8x4` に展開するため得るものが無く、H100 では 16x8x4 が最速（cuBLAS が選ぶ 16x8x8 ではない）。kK>4 を CUTLASS 2.x で正しく動かすための warp tile iterator も含む |
| [`tma_survey.md`](tma_survey.md) | TMA の適用可能性を候補ごとに実測した記録。採用ゼロだが、FP64 での受理条件・帯域・L1 挙動と、2 候補それぞれの構造的な不採用理由 |
| [`cublas_emulation_survey.md`](cublas_emulation_survey.md) | cuBLAS FP64 fixed-point emulation の見かけのストール調査。EAGER 強制、永続 8 GiB workspace、p=7/p=255 の速度と数値検証。**§6（2026-08-29）は p=1023 でも 1.74× 遅い理由**: ADP が 54 bit を選ぶ一方 FIXED 55 の x は 0.495×、y の 1024³ batch は FIXED 55 でも 1.41× 負け。INT8 算術は 64 倍足りている |
| [`ozaki2_survey_2504.08009.md`](ozaki2_survey_2504.08009.md) | arXiv:2504.08009v3（Ozaki Scheme II、INT8 Tensor Core による FP64 GEMM エミュレーション）の適用調査。不採用だが、成立条件が `p ≳ 500-650` であること、およびハードウェア条件が 3.82 FLOP/byte であることを実測から確定した |
| [`ozaki2_implementation_report.md`](ozaki2_implementation_report.md) | `feature/ozaki` の `CUDAFORTRAN_GEMM_OZAKI2` 本体実装。GEMMul8 参照実装との差分整理、moduli テーブル整合、数値・性能検証。性能結論は調査どおり不採用だが計測可能な経路として統合 |
| [`ozaki1_implementation_report.md`](ozaki1_implementation_report.md) | 同 worktree の `CUDAFORTRAN_GEMM_OZAKI1`（Ozaki Scheme I）。A/B 両スライス・最大 s² 本 INT8 GEMM・CRT なし FP64 加算。`scale_a` の z 方向バッファ修正を含む。p=7 で native 比 max abs ≈21（OZAKI2 と同オーダー）、device 時間は OZAKI2 より遅い典型 |
| [`p15_gap_study.md`](p15_gap_study.md) | p=7 と p=255 の間を同一 DOF で埋める最初の点 p=15 (Nq=16)。CUDA core 版と Tensor Core 版の融合カーネル、shared 戦略、Nq=16 では融合したまま占有率 50% を超えられないという構造的な壁。**§14（2026-08-27）は p=15 が同一 DOF の曲線から外れている**（p=31 が 1.83 倍の演算を 1.07 倍の時間でこなす）ことを追い、律速を測り直した: 命令数でも shared でも占有率でもなく **global ロード**（`long scoreboard` 48%、sector/request 13.46）。z の shared 往復を消しても **0%**、面 gather をタダにしても **−17.5% が上限**。p=63 で効いた「面 flux の別カーネル化」は面点率 37.5% の p=15 では 3 倍の赤字。**§15 で採用に至った**: 面フラックスに専用の shared バッファ 12 KB を与えて x パネル格納の直後に前倒しすると、**gather を 1 つも減らさないまま `long scoreboard` が 47.1% → 24.5% に半減し −5.3%**（345.1 → 326.8 µs/stage、ビット一致）。占有率はレジスタで決まっているので 12 KB は事実上タダ。**§16（2026-08-28）は「shared を節約する」という前提そのものを捨てて 332.0 → 272.0 µs/stage（−18.1%、ビット一致）**: カーブアウトの代金を先に測って **+64 KB まではタダ・+128 KB で崖**を確かめ、3 パネル同時 shared 化（バリア 8 → 3 本、−7.0%）、面フェーズの 2 面点/スレッド化（−3.0%）、`__restrict__`（−2.2%、**p=63 では −11.6%** と次数をまたぐ。p=7 だけは spill で +3.6% と逆効果）、**M 側 i 境界 2 面の shared 常駐**（§14.6 が残した唯一の筋、−6.1%）、z 往復のバリアを `__syncwarp` に（−0.8%）。床は 281.3 → 223.0 µs |
| [`p63_gap_study.md`](p63_gap_study.md) | 同一 DOF の 4 点目 p=63 (Nq=64)。任意次数の LGL 演算子生成、CUTLASS GEMM 経路の次数開放とその batch 上限、5 本のカーネルが 3 種類の理由で別々に詰まる様子、融合カーネルを書かない判断の根拠。§8 のその判断は 2026-08-27 に訂正され、**§13 で実際に両方書いて測った**: `FUSED` 970.7 µs / `FUSED_TC` 662.3 µs に対し `GEMM_FUSED` 598.6 µs で、**p=63 の最速は `GEMM_FUSED` のまま**。§8 の訂正注記が外挿した 365 µs は 1.8 倍外れており、理由は TC 版が帯域律速でも発行律速でもなく**レイテンシ律速**（DRAM 18%、占有率 24.6%）であること。**§16 でこれは覆った**: チャンクループを消し（`BK63` 16→64、動的 shared 96 KB）、shared レイアウトを直して **539.0 µs/stage、`GEMM_FUSED` の 587.3 に 1.090 倍**。**p=63 の最速は `CUDAFORTRAN_FUSED_TC` に交代**し、融合が勝つ上限は p=63 に上がった。§16.4 は「8 B の shared アクセスの条件は 32 レーンの `d mod 32` ではなく半ワープ 16 レーンの `d mod 16`」を SASS と実測で確定させている 。**§20（2026-08-28）で残り 3 ブロックの天井を測って探索終了**（面フラックスカーネルが実は 13.45% あり、その 79% は i 平面に連続方向が無いことによるギャザー増幅。契約内では直らない）。**§19（2026-08-28）で 487.8 µs/stage**（y カーネルを 2 ブロック/SM に、y エピローグの `dqdt` を `cp.async` で先読み、`sFU` のストアをコンフリクトフリーに。§17.1 の「構造的」は成立しなかった）。**§18（2026-08-28）で 477.2 µs/stage**: チャンクループ本体の末尾に置いていたバリアを先頭へ `if (kk)` 付きで移すと、`BK63 = NQ63` で 1 回しか回らないこのループでは末尾の 1 本がまるごと消える（ビット一致、−7.7%）。`GEMM_FUSED` との差は 1.101 倍から 1.20 倍に広がった。**§24（2026-08-30）は CUDA-core `FUSED` を 966.4 → 776.0 µs/stage（−19.7%）**: `__restrict__` とチャンクループ除去。計算 LDS を消すと −58% の天井があるが、`double4` / shuffle / `cp.async` は全部負け。主比 TC/FUSED は 2.29× → 1.84×。**`GEMM_FUSED` の x/y 重ねは −0.28%**。屋根には当たっていない |
| [`p127_gap_study.md`](p127_gap_study.md) | 同一 DOF の 5 点目 p=127 (Nq=128)。**演算強度がマシンバランスを越える最小の次数**。コード変更ゼロで 5 経路が通り、次数依存ノブ 4 種を掃引しても採用すべき変更が無かったこと、`K` が深くなると CUTLASS の x GEMM が cuBLAS に追いつくこと。**§11（2026-08-28）で `CUDAFORTRAN_FUSED` / `CUDAFORTRAN_FUSED_TC` を p=127 に開き、最終的に `FUSED_TC` 707.7 µs/stage で `GEMM_FUSED` の 711.3 を抜いて最速になった**（初稿の 1141 から 5 段階）。設計は融合 2 本とも 128×64 タイル・1024 スレッド・1 ワープ 2×2 mma タイル、**1 平面 1 回しか読まないフラックスパネルだけを全深さで常駐**させ 2 倍冗長な方と L2 常駐の `D1D` をチャンク。§11.9 の ncu が律速を確定させ（**Tensor Core 化は律速を shared から DMMA パイプへ移す**: CUDA core 版 mio throttle 13.1 / shared ld 201.3 M に対し TC 版 0.16 / 33.6 M）、§11.10 で**スウィズルは mma ループ内で恒等的に簡約でき**命令数 −29%、§11.11 で**チャンクループの末尾バリアが最終反復で無駄**と分かって −4.5%。§11.12 のモデル: **どの資源も飽和しておらず、5 種の削減のうちスケールしたのはバリアだけ** —— 律速は資源ではなく同期点である。**§12（同日）でこれは覆った**: `GEMM_FUSED` の z epilogue を 3 点直すと 789.8 → **752.8 µs/stage（−4.7%）**になり、**p=127 の最速は `CUDAFORTRAN_GEMM_FUSED` に戻った**（`FUSED_TC` 787.8 に 1.047 倍）。**§13（同日）でさらに 751.7 → 730.7 µs/stage（−2.8%、`Main` −3.0%）**: z GEMM は 5 本のストリームで epilogue のメモリ側に詰まっており、**134 MB 消しても 403 MB 消しても −72 µs で止まる**（= mainloop 床 139 µs に当たる）。そこで **`deriv_x` を y GEMM の 2 ソース epilogue に畳んで** z の読みを 1 本減らすと z −15.7 µs に対し y は +2.1 µs しか払わない（y は DMMA パイプ 81% / DRAM 18% で、しかも `deriv_x` は直前の x GEMM が書いたばかりで L2 に温かい）。lift の演算をアキュムレータの shared 往復の後ろへ回して −5 µs、epilogue のアクセスを 16 バイトにして −7.2 µs（32 バイトは赤字、`Nq<=64` では +2.8% なので `GemmZWide` を別型に）。**落とした候補**: ワープ数を増やす（x/y/z とも +2〜10%、発行スロットは元から余っている）、z のタイルとスウィズル（`flux_z` の L2 再利用は既に成立）、epilogue ループ展開。**§13.8 は計測衛生**: `++it_dy` を 2 行動かすだけでビット一致のまま +3.5% |
| [`p511_gap_study.md`](p511_gap_study.md) | p=511 (Nq=512, Ne=1) の `CUDAFORTRAN_GEMM` / `CUDAFORTRAN_GEMM_FUSED` 対応。packed halo allocation で主場6配列を host/device 各42→12 GiBへ削減。点変化係数を含む全134,217,728点の `dqdt` を最大絶対差3.55e-15で検証。GB200では `GEMM_FUSED` が13.159対13.528 ms/stageで2.73%速い |
| [`p575_gap_study.md`](p575_gap_study.md) | p=575 (Nq=576, Ne=1) の `CUDAFORTRAN_GEMM` / `CUDAFORTRAN_GEMM_FUSED` 対応。点変化係数を含む全191,102,976点の `dqdt` を最大絶対差3.55e-15で検証。GB200の3-run中央値では `GEMM_FUSED` が20.453対20.912 ms/stageで2.20%速い |
| [`p767_gap_study.md`](p767_gap_study.md) | p=767 (Nq=768, Ne=1) の `CUDAFORTRAN_GEMM` / `CUDAFORTRAN_GEMM_FUSED` 対応。点変化係数を含む全452,984,832点の `dqdt` を最大絶対差3.55e-15で検証。GB200の3-run中央値では `GEMM_FUSED` が60.362対62.817 ms/stageで3.91%速い |
| [`p1023_gap_study.md`](p1023_gap_study.md) | p=1023 (Nq=1024, Ne=1) の `CUDAFORTRAN_GEMM` / `CUDAFORTRAN_GEMM_FUSED` 対応。Escale方向offsetだけを64-bit安全化し、未使用surfaceとz中間配列を除いた。正確な配列payloadは144.320 GiBだが、OpenACC allocator込みの実測peak増分はGEMM 176.416 GiB / FUSED 176.358 GiB。p=511/575/767/1023実測を覆う事前見積もりを`payload*1.25+2 GiB`とした。点変化係数を含む全1,073,741,824点を最大絶対差3.55e-15で検証。GB200では `GEMM_FUSED` が187.617対194.058 ms/stageで3.32%速い |
| [`index64_boundary_validation.md`](index64_boundary_validation.md) | 高次数の host-side extent / pointer offset を64-bit安全化。p=7/15/511 の全 owned `dqdt` は変更前後でビット一致、device SASSも一致。性能差は −0.19%〜+0.02%で既存経路への影響なし |
| [`p31_gap_study.md`](p31_gap_study.md) | 同一 DOF の 6 点目にして最後の点 p=31 (Nq=32)。**最速は `CUDAFORTRAN_FUSED_TC`**（§14、374.8 µs/stage；§16 device 359.7）。**§18–19（2026-08-30）は残り天井を測って探索終了**：xz は `lg_throttle`、y は `mio_throttle`。面 2,4 の天井 −17.8% に対し実装 4 形は +25.6% / +25.6% / ±0 / +31.5%。占有率 50% はスピルまたは `lg_throttle` 増で +25〜29%。採用ゼロ。Nq=32 の Tensor Core 融合カーネル 2 本で CUDA core 融合版の **2.66 倍**、`GEMM` の 1.67 倍。x と z が同じ出力写像を共有するので z の shared 往復が無く、転置形にすると D1D がレジスタに載る。**「p=31 は曲線の極大点」「融合が勝つ上限は p=15」という当初の結論はこれで否定された**（§14.2、§14.1 に訂正注記）。Nq=32 の CUDA core 融合カーネル 2 本、CUTLASS 経路は p=31 で使えるという訂正、**lift と assembly の融合で `GEMM` / `GEMM_CUTE` 経路を全次数 1 割速く**した（p=31 −11.7%、p=63 −12.0%、p=127 −10.2%、ビット一致）。**§20（2026-08-30）は CC 融合 `FUSED` を 992.5 → 718.2 µs/stage（−27.6%、ビット一致）**: 512 スレッド `double2` + スラブ、xz 平面の先読み、パネル二重バッファ。論文の主比は 2.78× → **2.00×** |

## 経路の役割（2026-08-29 以降）

最速を取りに行く本番経路は次数ごとの `CUDAFORTRAN_GEMM_FUSED` または
`CUDAFORTRAN_FUSED_TC` である。融合経路は論文用に 3 つある。GB200 では
FP64 Tensor Core ピークが CUDA core ピークと同じ（40.1 TFLOP/s）なので、
iso-schedule の DFMA 置換は屋根の引き上げではない。

固定するものと測るもの:

- **B. `CUDAFORTRAN_FUSED` 対 `CUDAFORTRAN_FUSED_TC`**: 同じ数値契約の融合。
  前者は CUDA core 向けスケジュール（自然順 shared、長さ `Nq` の内積）、
  後者は本番 Tensor Core。論文の主比。`FUSED` は独立に最速レースしない。
  namelist `FUSED` は CC カーネルだけを起動する。`FUSED_DFMA` で代行しない。
- **A. `CUDAFORTRAN_FUSED_TC` 対 `CUDAFORTRAN_FUSED_DFMA`**: 同一
  `cuda_dg_kernels_tc.cu` で内積だけ MMA / DFMA。メカニズム節用。DFMA は
  独立に最速化しない。

- `CUDAFORTRAN_GEMM_CUTE`: `GEMM_FUSED` と同じ CUTLASS volume GEMM 本体
  （`Nq<=64` では x を cuBLAS）で、融合エピローグが無い版。`GEMM` との差は
  ライブラリ、`GEMM_FUSED` との差は融合パッケージ。

`CUDAFORTRAN_GEMM_OZAKI1` / `_OZAKI2` は未融合 `GEMM` と同じ周囲で volume
GEMM だけを差し替える。cuBLAS native および `CublasEmulation` との比較用。

下の「現時点の結論」と各レポートの表は書かれた時点の値のまま残す。

### 2026-08-29 の `FUSED` 行は iso-schedule DFMA だった

当時の namelist `CUDAFORTRAN_FUSED` は Fortran カーネルではなく
`cuda_dg_kernels_tc.cu` の `UseTc=false` だった。その日のまとめ表の値は
経路名を **`CUDAFORTRAN_FUSED_DFMA` に付け替えて残す**（証拠を消さない）。
旧 Fortran / 専用 CC の表（p=7 device 〜324 µs など）は CC 最適の旧測として
残す。login ノード、`conf_perf_p63_fused.conf` の 3-run 中央値は
Main **2.767 ms/step**、device fused **48.30 ms**（19 measured steps）で、
実体は DFMA。同一 conf の `FUSED_TC` は Main 1.500 ms/step、device 23.87 ms。
点変化係数では p=7/15/31/63/127/255 で当時の FUSED（いまの DFMA）と
FUSED_TC の `dqdt` はビット一致。

`GEMM_CUTE` / `GEMM_FUSED` は `fuse_epilogue` 付きの単一ドライバ。HEAD
（`fd091fc`）と同じ `conf_perf_p63_gemm_fused.conf` / `conf_perf_p63_tc.conf`
で測り直すと、device 時間の中央値差は GEMM_FUSED **−0.3%**、FUSED_TC **−0.3%**
で測定誤差の範囲。本番カーネルのタイルと MMA ループは動かしていない。

## 最新結果のまとめ（2026-08-29、p=31 `FUSED` のみ 2026-08-30）

GB200 1 GPU（login node GPU 1）、`make CUDA=1 GPUFLAGS=-gpu=cc100`、
`UseCudaGraph = .false.`。p=7…255 の DFMA / TC / GEMM / SPLIT 行は
**2026-08-29 に同一実行ファイルで 3-run 中央値を採り直した値**（各次数の既存
`conf_perf_p*` から `DqdtKernel_Type` だけを差し替え。`Ne` / `dt` / `nstep` は
変えていない）。`CUDAFORTRAN_FUSED`（CUDA-core 融合）の p=15 / 63 / 127 / 255
行は CC カーネルを戻した作業ツリー（親 `959ad50`）で同じ conf を 3-run した値。
**p=31 の `FUSED` 行だけは [`p31_gap_study.md`](p31_gap_study.md) §20**（job
`69623`、12 回交互 A/B、718.2 µs）。p=7 の CC 行は同日の復活測定（326.8 µs）。
p≥511 は各 gap study の測定のまま（再実行していない）。
次数別の再測定節: p=7 は [`execution_times.md`](execution_times.md) 追記 16・18、
p=15 §18–19、p=31 §16–20、p=63 §22–23、p=127 §15–16、p=255 §12–15.11。

太字は次数ごとの最速（Main ms/step）。空欄はその次数では経路が無い
（`GEMM_FUSED` / `GEMM_CUTE` は `Nq*Ne <= 65535`、p≥511 は
`GEMM` / `GEMM_FUSED` のみ）。`CUDAFORTRAN_GEMM_OZAKI1` / `_OZAKI2` は
本番最速ではないので表に入れない。

p=7…255 は同一体積 DOF（`NeX*Nq = 256`、16,777,216 点）。p≥511 は `Ne=1`。

**µs/stage のタイマー**: CUDA 経路は実行ファイルの `CUDA device *`（SPLIT は
4 本の device 時間 + `Element boundary flux`）。OpenACC は
`Volume derivate + surface lift` + `Element boundary flux`。
以前の表の一部は `Cal_tend`（launch 込み）だったので、同じカーネルでも
数 %〜十数 % 遅めに出ていた。FLOP/s と TB/s はすべてこの µs/stage で割る。
Main ms/step は `Main per step:`（RK 更新と halo を含む）。

### カウントの定義（全経路共通の FLOP、2 種類の DRAM）

FLOP と unique DRAM は実装に依らない。パス列の DRAM だけ経路で変わる。

**FLOP / RK stage**（全パス同一。multiply/add = 1、FMA = 2。ncu の発行命令ではない）:

- 体積: `(6*Nq + 20) * Np * Ne`（3 方向の縮約 `3*2*Nq`、流束積 3、`Escale` / lift / 和で 20）
- 面: `21 * 6 * Nq² * Ne`（数値流束 1 面点あたり 21）
- SSP-RK3 なので 1 step の演算量は 3 stage 分。下の TFLOP/s は
  **tendency 1 stage の時間**（µs/stage）で割った値。Main には RK 更新と halo が乗る。
- ピーク分母は GB200 FP64 **40.1 TFLOP/s**（CUDA core と Tensor Core で同じ）。

**DRAM unique / RK stage**（全パス同一。「各配列を 1 RK stage で一度だけ読み書き」）:

- 体積 8 本: `q, u, v, w` 読み、`Escale` 3 方向読み、`dqdt` 書き → `64 B/node`
- 面だけの入力: `normal_fn` 3 成分 + `Fscale` + `VMapM`/`VMapP`（int32×2）
  → `40 B/face-pt` = `240/Nq B/node`
- 合計 `64 + 240/Nq` B/node。同一点を体積と面で二度数えない。
- RK 更新（`q0`/`q`/`dqdt`）は tendency に含めない。

**DRAM path / RK stage**（経路ごとのアルゴリズム上の R/W。L2 ヒットも数える）:

| 経路 | 体積 | 面 |
|---|---|---|
| `OPENACC_ASIS` | unique と同じ（要素ローカル一時は DRAM に出さない） | unique と同じ |
| `FUSED` / `FUSED_DFMA` / `FUSED_TC` | unique の 8 本 | unique の面入力に加え、M/P の `q,u,v,w` gather `64 B/face-pt` |
| `GEMM` / `GEMM_CUTE` | `volume_flux` 7V + 3 GEMM 6V + assembly 7V = **20V**（160 B/node） | `elembnd` 112 B/face-pt |
| `GEMM_FUSED` | 中間を 1 本畳んだ **18V**（144 B/node） | 同上 112 B/face-pt |
| `OPENACC_SPLIT` / `CUDAFORTRAN_SPLIT` | flux 7V + deriv 6V + lift_out 1V + assembly 8V = 22V。`Nq<=128` では dense `Lift_mat` 追加 6V | 112 B/face-pt |

`V = 8*Np*Ne`。112 B/face-pt は VMap 8 + `q,u,v,w`×M/P 64 + 法線 24 + `Fscale` 8 + `flux_bnd` 書き 8。

path 側の TB/s は **モデル転送量 ÷ 時間**であり、DRAM ピーク 7.9 TB/s の達成率ではない。
p=7 `FUSED_TC` のように gather が L2 に載る次数では、path 値がピークを超えて見える。
DRAM 屋根との比較は unique 列を使う。

今回埋めた欠測・訂正した不自然な値:

- p=255 の Main は µs/stage からの換算をやめ、実測した。`FUSED` と `GEMM_CUTE` を追加。
- p=15 `GEMM` の 2508 µs/stage は、device 合計を RK 3 stage で割っていなかった。実測は **848 µs**。
- p=7 `GEMM` の Main 9.27 ms は RK 最適化前の `nstep=1000` 値。同一 conf では **5.09 ms**。
- SPLIT / OpenACC の µs/stage（† だった行）を実測した。
- p=7 の C++ `FUSED_DFMA`（当時 namelist `FUSED`、`UseTc=false`）は旧 Fortran
  融合（device 〜324 µs）より遅く **428 µs**。TC 版との iso-schedule 対照は
  同一 C++ ソースに限る。旧 Fortran の 324 µs は CC 最適の旧測として残す。

### 仕事量（パス非依存）

| p | Nq | Ne | 体積点 | FLOP / RK stage | unique DRAM / RK stage | B/node |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 8 | 32³ | 16,777,216 | 1.405e9 | 1577 MB | 94.00 |
| 15 | 16 | 16³ | 16,777,216 | 2.078e9 | 1325 MB | 79.00 |
| 31 | 32 | 8³ | 16,777,216 | 3.623e9 | 1200 MB | 71.50 |
| 63 | 64 | 4³ | 16,777,216 | 6.811e9 | 1137 MB | 67.75 |
| 127 | 128 | 2³ | 16,777,216 | 1.324e10 | 1105 MB | 65.88 |
| 255 | 256 | 1 | 16,777,216 | 2.611e10 | 1089 MB | 64.94 |
| 511 | 512 | 1 | 134,217,728 | 4.150e11 | 8653 MB | 64.47 |
| 575 | 576 | 1 | 191,102,976 | 6.643e11 | 12310 MB | 64.42 |
| 767 | 768 | 1 | 452,984,832 | 2.097e12 | 29133 MB | 64.31 |
| 1023 | 1024 | 1 | 1,073,741,824 | 6.619e12 | 68971 MB | 64.23 |

path 側の B/node（上表の unique 列と対になる）:

| p | unique | FUSED / DFMA / TC | GEMM / CUTE | GEMM_FUSED | SPLIT |
|---:|---:|---:|---:|---:|---:|
| 7 | 94.0 | 142.0 | 244.0 | — | 308.0 |
| 15 | 79.0 | 103.0 | 202.0 | — | 266.0 |
| 31 | 71.5 | 83.5 | 181.0 | 165.0 | 245.0 |
| 63 | 67.8 | 73.8 | 170.5 | 154.5 | 234.5 |
| 127 | 65.9 | 68.9 | 165.2 | 149.2 | 229.2 |
| 255 | 64.9 | 66.4 | 162.6 | 146.6 | — |
| 511 | 64.5 | — | 161.3 | 145.3 | — |
| 575 | 64.4 | — | 161.2 | 145.2 | — |
| 767 | 64.3 | — | 160.9 | 144.9 | — |
| 1023 | 64.2 | — | 160.7 | 144.7 | — |

### 経路 × 次数

ピーク比の分母は 40.1 TFLOP/s と 7.9 TB/s。

#### p=7（`Ne=32³`、`nstep=20`）

| 経路 | ms/step | µs/stage | TFLOP/s | vs 40.1 | TB/s unique | vs 7.9 | TB/s path | vs 7.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `OPENACC_ASIS` | 3.492 | 1065.8 | 1.32 | 3.3% | 1.48 | 18.7% | 1.48 | 18.7% |
| `OPENACC_SPLIT` | 2.708 | 807.8 | 1.74 | 4.3% | 1.95 | 24.7% | 6.40 | 81.0% |
| `CUDAFORTRAN_SPLIT` | 2.565 | 764.1 | 1.84 | 4.6% | 2.06 | 26.1% | 6.76 | 85.6% |
| `CUDAFORTRAN_FUSED` | 1.227 | 326.8 | 4.30 | 10.7% | 4.83 | 61.1% | 7.29 | 92.3% |
| `CUDAFORTRAN_FUSED_DFMA` | 1.528 | 427.8 | 3.28 | 8.2% | 3.69 | 46.7% | 5.57 | 70.5% |
| **`CUDAFORTRAN_FUSED_TC`** | **1.073** | **274.9** | **5.11** | **12.7%** | **5.74** | **72.6%** | 8.67 | 110% |
| `CUDAFORTRAN_GEMM` | 5.088 | 1635.4 | 0.86 | 2.1% | 0.96 | 12.2% | 2.50 | 31.7% |

`GEMM_FUSED` / `GEMM_CUTE` は `Nq*Ne = 262144 > 65535` のため無し。
path 列 110% は面 gather をアルゴリズム通り数えた結果で、実 DRAM は unique の 72.6%。
`CUDAFORTRAN_FUSED` は CUDA core 向け融合の復活（login GPU 1、3-run 中央値、
device 326.8 µs）。`FUSED_DFMA` の 427.8 µs は 2026-08-29 の iso-schedule 測定のまま。
論文の主比は **TC / FUSED = 274.9 / 326.8 = 1.19×**。メカニズム節の
**TC / FUSED_DFMA = 274.9 / 427.8 = 1.56×** は MMA 命令の有無だけ。

#### p=15（`Ne=16³`、`nstep=20`）

| 経路 | ms/step | µs/stage | TFLOP/s | vs 40.1 | TB/s unique | vs 7.9 | TB/s path | vs 7.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `OPENACC_SPLIT` | 3.335 | 1015.1 | 2.05 | 5.1% | 1.31 | 16.5% | 4.40 | 55.7% |
| `CUDAFORTRAN_SPLIT` | 5.468 | 1747.1 | 1.19 | 3.0% | 0.76 | 9.6% | 2.55 | 32.3% |
| `CUDAFORTRAN_FUSED` | 1.583 | 446.7 | 4.65 | 11.6% | 2.97 | 37.6% | 3.87 | 49.0% |
| `CUDAFORTRAN_FUSED_DFMA` | 1.583 | 446.6 | 4.65 | 11.6% | 2.97 | 37.6% | 3.87 | 49.0% |
| **`CUDAFORTRAN_FUSED_TC`** | **1.068** | **271.8** | **7.65** | **19.1%** | **4.88** | **61.7%** | 6.36 | 80.5% |
| `CUDAFORTRAN_GEMM` | 2.766 | 847.5 | 2.45 | 6.1% | 1.56 | 19.8% | 4.00 | 50.6% |

`GEMM_FUSED` / `GEMM_CUTE` は `Nq*Ne = 65536` でバッチ上限ちょうど外。
`CUDAFORTRAN_FUSED` は CC 復活（login GPU 1、3-run 中央値）。device 446.7 µs は
iso-schedule DFMA の 446.6 と測定誤差内。これは同一カーネルの二重測定ではない
（点変化係数の `dqdt` は FUSED と DFMA で 1 ulp、DFMA と TC はビット一致。
独立再測は [`execution_times.md`](execution_times.md) 追記 20）。この次数では
論文の主比 B とメカニズム比 A が同じ数字になる。
**TC / FUSED = 271.8 / 446.7 = 1.64×**。

#### p=31（`Ne=8³`、`nstep=200`）

| 経路 | ms/step | µs/stage | TFLOP/s | vs 40.1 | TB/s unique | vs 7.9 | TB/s path | vs 7.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CUDAFORTRAN_SPLIT` | 8.196 | 2634.6 | 1.38 | 3.4% | 0.46 | 5.8% | 1.56 | 19.7% |
| `CUDAFORTRAN_FUSED` | 2.415 | 718.2 | 5.05 | 12.6% | 1.67 | 21.1% | 1.95 | 24.7% |
| `CUDAFORTRAN_FUSED_DFMA` | 2.634 | 790.1 | 4.59 | 11.4% | 1.52 | 19.2% | 1.77 | 22.4% |
| **`CUDAFORTRAN_FUSED_TC`** | **1.342** | **359.7** | **10.07** | **25.1%** | **3.33** | **42.2%** | 3.89 | 49.3% |
| `CUDAFORTRAN_GEMM` | 2.139 | 625.7 | 5.79 | 14.4% | 1.92 | 24.3% | 4.85 | 61.4% |
| `CUDAFORTRAN_GEMM_CUTE` | 2.685 | 808.6 | 4.48 | 11.2% | 1.48 | 18.8% | 3.76 | 47.5% |
| `CUDAFORTRAN_GEMM_FUSED` | 3.083 | 940.6 | 3.85 | 9.6% | 1.28 | 16.1% | 2.94 | 37.3% |

この次数だけ `GEMM_FUSED` が素の `GEMM` より遅い（浅い `K=32` で z epilogue 融合の方が高い）。
`CUDAFORTRAN_FUSED` は [`p31_gap_study.md`](p31_gap_study.md) §20 の CC 融合
（device **718.2 µs**、job `69623`）。§17 の 998.4 µs は改修前。論文の主比は
**TC / FUSED = 359.7 / 718.2 = 2.00×**。メカニズム比
**TC / FUSED_DFMA = 359.7 / 790.1 = 2.20×** は変わらない。改修後の CC は
iso-schedule DFMA より速い（790.1 / 718.2 = 1.10×）。

#### p=63（`Ne=4³`、`nstep=20`）

| 経路 | ms/step | µs/stage | TFLOP/s | vs 40.1 | TB/s unique | vs 7.9 | TB/s path | vs 7.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CUDAFORTRAN_SPLIT` | 12.01 | 3967.2 | 1.72 | 4.3% | 0.29 | 3.6% | 0.99 | 12.6% |
| `CUDAFORTRAN_FUSED` | 2.110 | 618.6 | 11.01 | 27.5% | 1.84 | 23.3% | 2.01 | 25.4% |
| `CUDAFORTRAN_FUSED_DFMA` | 2.766 | 846.2 | 8.05 | 20.1% | 1.34 | 17.0% | 1.46 | 18.5% |
| **`CUDAFORTRAN_FUSED_TC`** | **1.512** | **421.7** | **16.15** | **40.3%** | **2.70** | **34.1%** | 2.93 | 37.1% |
| `CUDAFORTRAN_GEMM` | 2.102 | 622.0 | 10.95 | 27.3% | 1.83 | 23.1% | 4.60 | 58.2% |
| `CUDAFORTRAN_GEMM_CUTE` | 2.144 | 636.6 | 10.70 | 26.7% | 1.79 | 22.6% | 4.49 | 56.9% |
| `CUDAFORTRAN_GEMM_FUSED` | 2.139 | 634.6 | 10.73 | 26.8% | 1.79 | 22.7% | 4.08 | 51.7% |

以前の `FUSED_TC` 487.8 µs は `Cal_tend`。device 時間は 422 µs。`GEMM` と
`GEMM_FUSED` は 2% 以内で、最速は `FUSED_TC` のまま。
`CUDAFORTRAN_FUSED` は [`p63_gap_study.md`](p63_gap_study.md) §25 の CC
（512 スレッド × 8 連続 k、job `69651`）。§24 の 776.0 µs と §23 の 966.5 µs は
その時点の測定。論文の主比は **TC / FUSED = 421.7 / 618.6 = 1.47×**。

#### p=127（`Ne=2³`、`nstep=100`）

| 経路 | ms/step | µs/stage | TFLOP/s | vs 40.1 | TB/s unique | vs 7.9 | TB/s path | vs 7.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CUDAFORTRAN_SPLIT` | 20.99 | 6916.1 | 1.91 | 4.8% | 0.16 | 2.0% | 0.56 | 7.0% |
| `CUDAFORTRAN_FUSED` | 5.033 | 1593.5 | 8.31 | 20.7% | 0.69 | 8.8% | 0.73 | 9.2% |
| `CUDAFORTRAN_FUSED_DFMA` | 6.660 | 2137.3 | 6.19 | 15.4% | 0.52 | 6.5% | 0.54 | 6.8% |
| `CUDAFORTRAN_FUSED_TC` | 2.380 | 706.1 | 18.75 | 46.8% | 1.57 | 19.8% | 1.64 | 20.7% |
| `CUDAFORTRAN_GEMM` | 2.473 | 737.3 | 17.95 | 44.8% | 1.50 | 19.0% | 3.76 | 47.6% |
| `CUDAFORTRAN_GEMM_CUTE` | 2.508 | 749.6 | 17.66 | 44.0% | 1.47 | 18.7% | 3.70 | 46.8% |
| **`CUDAFORTRAN_GEMM_FUSED`** | **2.283** | **674.0** | **19.64** | **49.0%** | **1.64** | **20.8%** | 3.71 | 47.0% |

`CUDAFORTRAN_FUSED` は旧 Fortran CC（〜1585 µs）と同水準で、iso-schedule DFMA
より速い。論文の主比は **TC / FUSED = 706.1 / 1593.5 = 2.26×**。
点変化係数の owned `dqdt` は性能と同じ `Ne=2³`（16,777,216 点）で FUSED /
DFMA / TC が全点ビット一致（[`p127_gap_study.md`](p127_gap_study.md) §16）。

#### p=255（`Ne=1`、`nstep=20`）

| 経路 | ms/step | µs/stage | TFLOP/s | vs 40.1 | TB/s unique | vs 7.9 | TB/s path | vs 7.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CUDAFORTRAN_FUSED` | 5.774 | 1868.6 | 13.94 | 34.8% | 0.58 | 7.4% | 0.58 | 7.4% |
| `CUDAFORTRAN_FUSED_DFMA` | 6.160 | 1998.0 | 13.07 | 32.6% | 0.55 | 6.9% | 0.56 | 7.1% |
| **`CUDAFORTRAN_FUSED_TC`** | **2.978** | **918.9** | **28.42** | **70.9%** | **1.19** | **15.0%** | 1.21 | 15.4% |
| `CUDAFORTRAN_GEMM` | 3.441 | 1075.7 | 24.28 | 60.5% | 1.01 | 12.8% | 2.54 | 32.1% |
| `CUDAFORTRAN_GEMM_CUTE` | 3.432 | 1072.9 | 24.34 | 60.7% | 1.02 | 12.9% | 2.54 | 32.2% |
| `CUDAFORTRAN_GEMM_FUSED` | 3.110 | 963.4 | 27.11 | 67.6% | 1.13 | 14.3% | 2.55 | 32.3% |

`CUDAFORTRAN_FUSED` は 16×16 タイルの CC。Fortran `2dadc41^` に `launch_bounds`
は無く、C++ の `__launch_bounds__(256,1)` がレジスタを溜めて 5252.8 µs だった。
`minBlocks=8` で 5058.4 µs（§14）。2 点/スレッドは 4684.4 µs（§15.3）。
y/z のフラックス再利用で 11.167 ms / 3697.3 µs（§15.5）。
x の D を `__ldg` で 10.821 ms / 3579.6 µs（§15.8）。
**y/z の `sD` を `{D0,D1}` の `double2` にすると Main 10.723 ms / 3546.3 µs/stage**
（§15.12–15.13、占有 GPU では 3585.5 → 3549.4 µs）。
x エピローグの Escale を `__ldg` すると login **10.713 ms / 3543.9 µs**（§15.39、
占有 −0.05%）。
**x の l タイル 16→32 で login 10.686 ms / 3535.1 µs**（§15.49、占有 3547.1 →
3538.3 µs、**−0.25%**）。
**z が 2 line タイルで D を dual sQ 同時内積すると login 9.927 ms / 3276.9 µs**
（§15.58、占有 3538.5 → 3279.3 µs、**−7.3%**）。
**y も 2 `tile_i` で dual sQ にすると login 9.767 ms / 3223.6 µs**（§15.59、
占有 3279.0 → 3227.6 µs、**−1.6%**）。
**x も 2 `tile_j` で dual sQ（D は `__ldg` のまま）にすると login 9.496 ms /
3130.8 µs**（§15.60、占有 3227.6 → 3131.4 µs、**−3.0%**）。
z の 4 line 同時内積は +21.9% で不採用（§15.61）。
x の D をワープ内 shfl すると +28.9%（§15.62）。
**x の `sQ` 行ストライド 33 で login 9.392 ms / 3095.6 µs**（§15.63、占有
3136.5 → 3098.7 µs、**−1.2%**）。ストライド 48 は +1.6%（§15.64）。
**x を 2 `j` × 2 `i` に組み替えると login 8.965 ms / 2950.9 µs**（§15.68、占有
3088.4 → 2945.2 µs、**−4.6%**）。4 `j` のまま 2 `i` を足すと +34.7%（§15.67）。
**y の 4 `tile_i` 同時内積で login 8.900 ms / 2929.0 µs**（§15.69、占有
2953.2 → 2929.0 µs、**−0.8%**）。x の 4 `j` × 2 `i` は +41.3%（§15.70）。
**y だけ `launch_bounds(128, 8)` で login 7.902 ms / 2590.3 µs**（§15.74、占有
2926.4 → 2596.7 µs、**−11.3%**）。32 レジスタ天井の local スピルが消える。
**x の 4`j`×2`i` を同じ 64 レジスタ予算で載せると login 7.272 ms / 2376.0 µs**
（§15.75、占有 2595.5 → 2375.3 µs、**−8.5%**）。§15.70 の +41% はレジスタ不足。
**z の 4 line も同じ予算で login 7.205 ms / 2353.6 µs**（§15.76、占有
2374.5 → 2354.2 µs、**−0.9%**）。§15.61 の +21.9% もレジスタ不足。
y `minBlocks=12` は +3.1%（§15.77）。
**y を 4 `j` × 2 `i` にして Q を共有すると login 6.689 ms / 2178.9 µs**（§15.80、
占有 2352.7 → 2184.0 µs、**−7.2%**）。
**z を 4 `k` × 2 line にして Q を共有すると login 5.800 ms / 1877.2 µs**（§15.81、
占有 2181.1 → 1880.8 µs、**−13.8%**）。
**最終タイルの末尾バリア省略で login 5.791 ms / 1874.2 µs**（§15.89、占有
1879.3 → 1874.9 µs、**−0.23%**）。
**y の sQ/sD 二重バッファで login 5.774 ms / 1868.6 µs**（§15.92、占有
1874.6 → 1869.3 µs、**−0.28%**）。
y の sQ を同じ形に畳むと +0.07% で不採用（§15.15、job 67772）。
論文の主比は **TC / FUSED = 918.9 / 1868.6 = 2.04×**。
5252.8 / 5058.4 / 4684.4 / 3697.3 / 3579.6 µs は gap study に残す。

#### p=511 / 575 / 767 / 1023（`Ne=1`、各 gap study の値。今回は再測定していない）

| p | 経路 | ms/step | µs/stage | TFLOP/s | vs 40.1 | TB/s unique | vs 7.9 | TB/s path | vs 7.9 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 511 | `CUDAFORTRAN_GEMM` | 41.80 | 13528 | 30.68 | 76.5% | 0.64 | 8.1% | 1.60 | 20.3% |
| 511 | **`CUDAFORTRAN_GEMM_FUSED`** | **40.70** | **13159** | **31.54** | **78.7%** | 0.66 | 8.3% | 1.48 | 18.8% |
| 575 | `CUDAFORTRAN_GEMM` | 64.96 | 20912 | 31.77 | 79.2% | 0.59 | 7.5% | 1.47 | 18.6% |
| 575 | **`CUDAFORTRAN_GEMM_FUSED`** | **63.58** | **20453** | **32.48** | **81.0%** | 0.60 | 7.6% | 1.36 | 17.2% |
| 767 | `CUDAFORTRAN_GEMM` | 192.2 | 62817 | 33.37 | 83.2% | 0.46 | 5.9% | 1.16 | 14.7% |
| 767 | **`CUDAFORTRAN_GEMM_FUSED`** | **185.0** | **60362** | **34.73** | **86.6%** | 0.48 | 6.1% | 1.09 | 13.8% |
| 1023 | `CUDAFORTRAN_GEMM` | 578.9 | 194058 | 34.11 | 85.1% | 0.36 | 4.5% | 0.89 | 11.3% |
| 1023 | **`CUDAFORTRAN_GEMM_FUSED`** | **562.5** | **187617** | **35.28** | **88.0%** | 0.37 | 4.7% | 0.83 | 10.5% |

### 最速経路だけ横断

| p | 最速 | ms/step | µs/stage | TFLOP/s | vs 40.1 | TB/s unique | vs 7.9 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | `FUSED_TC` | 1.073 | 274.9 | 5.11 | 12.7% | 5.74 | 72.6% |
| 15 | `FUSED_TC` | 1.068 | 271.8 | 7.65 | 19.1% | 4.88 | 61.7% |
| 31 | `FUSED_TC` | 1.342 | 359.7 | 10.07 | 25.1% | 3.33 | 42.2% |
| 63 | `FUSED_TC` | 1.512 | 421.7 | 16.15 | 40.3% | 2.70 | 34.1% |
| 127 | `GEMM_FUSED` | 2.283 | 674.0 | 19.64 | 49.0% | 1.64 | 20.8% |
| 255 | `FUSED_TC` | 2.978 | 918.9 | 28.42 | 70.9% | 1.19 | 15.0% |
| 511 | `GEMM_FUSED` | 40.70 | 13159 | 31.54 | 78.7% | 0.66 | 8.3% |
| 575 | `GEMM_FUSED` | 63.58 | 20453 | 32.48 | 81.0% | 0.60 | 7.6% |
| 767 | `GEMM_FUSED` | 185.0 | 60362 | 34.73 | 86.6% | 0.48 | 6.1% |
| 1023 | `GEMM_FUSED` | 562.5 | 187617 | 35.28 | 88.0% | 0.37 | 4.7% |

同一 DOF（p=7…255）では次数が上がるほど unique 転送は面点率 `6/Nq` で減り、
FLOP は `6*Nq` で増える。達成 FLOP 比は 13% → 71% と単調に上がり、unique
帯域比は 73% → 15% と下がる。p≥511 は点が 8〜64 倍なので Main は ms から
数百 ms へ跳ね、演算屋根（76–88%）に張り付く。

p=7…255 の一次ソースは 2026-08-29 の login-node 再測定（上表）。**例外は
p=31 の `FUSED` 行だけ**で、[`p31_gap_study.md`](p31_gap_study.md) §20
（job `69623`）の 718.2 µs である。gap study の当時値（特に `Cal_tend`
ベースの µs/stage）は書き換えない。p≥511 は `p511_gap_study.md` ほか §性能。

## 現時点の結論

- **p=63 `GEMM_FUSED` の z 体積ロード移動（2026-08-30、`p63_gap_study.md` §42）**:
  device **+0.66%**（z 152→155 µs）。lift-behind の隙間に体積ロードを入れると損。

- **p=63 `GEMM_FUSED` の契約内ノブ（2026-08-30、`p63_gap_study.md` §41）**:
  flux DRAM 屋根・z 占有・y 隠し・persist 隔離まで測って閉じた。残るのは
  RK 融合と配列パックで、どちらも範囲外。ベースライン 571 µs/stage。最速は
  `FUSED_TC`。

- **p=63 `GEMM_FUSED` の加重 y 5 stage（2026-08-30、`p63_gap_study.md` §40）**:
  device **+0.64%**（y 130→134 µs）。stage 軸は 4 が谷。

- **p=63 `GEMM_FUSED` の GEMM 限定 persist（2026-08-30、`p63_gap_study.md` §39）**:
  flux 前に `SetLimit(0)` しても device **+16.7%**。flux は 128 µs のまま、y が
  130→145 µs。max persist 予算が GEMM の L2 を削る。

- **p=63 `GEMM_FUSED` の z `launch_bounds`（2026-08-30、`p63_gap_study.md` §38）**:
  8 CTA/SM 強制は device **+90%**（z 152→658 µs）。spill が占有の賞金を上回る。

- **p=63 `GEMM_FUSED` の z MaxShared carveout（2026-08-30、`p63_gap_study.md` §37）**:
  device **+8.0%**（z 152→196 µs）。L1 を削るとエピローグが伸びる。§33 と合わせ
  carveout の両極は既定より遅い。

- **p=63 `GEMM_FUSED` の加重 y warp `<32,64>`（2026-08-30、`p63_gap_study.md` §36）**:
  device **+47%**。y が 130→398 µs。加重エピローグは 4 ワープが必要。

- **p=63 `GEMM_FUSED` の加重 y 3 stage（2026-08-30、`p63_gap_study.md` §35）**:
  device **+0.53%**（y 130→133 µs）。折り込み後も 4 stage が勝つ。

- **p=63 `GEMM_FUSED` の y side2 最高優先度（2026-08-30、`p63_gap_study.md` §34）**:
  device **+0.03%**、レンジ重複。y と cuBLAS x の同居は優先度では動かない。

- **p=63 `GEMM_FUSED` の z MaxL1 carveout（2026-08-30、`p63_gap_study.md` §33）**:
  device **+51.8%**（571 → 867 µs/stage）。z assembly が 130–152 → **442 µs**。
  CUTLASS の shared タイルを L1 に奪う。占有 12% はレジスタ律速のまま。

- **p=63 `GEMM_FUSED` の 32 KB persist（2026-08-30、`p63_gap_study.md` §32）**:
  容量だけでも窓付きでもレンジ重複。`D1D_tr` は既に L2 に載っている。

- **p=63 `GEMM_FUSED` の `D1D_tr` persist（2026-08-30、`p63_gap_study.md` §31）**:
  Normal miss でも device **+58%**。`SetLimit` が通常 L2 を削り次の `volume_flux`
  が 128→318 µs。max 容量の persist はこの経路では使えない。

- **p=63 `GEMM_FUSED` の persistent flux（2026-08-30、`p63_gap_study.md` §30）**:
  CTA 4096 の grid-stride は device **+1.35%**（flux 128→136 µs）。パイプが細る。

- **p=63 `GEMM_FUSED` の flux CTA / TMA / 優先度（2026-08-30、`p63_gap_study.md` §29）**:
  CTA 128/512 と side 最低優先度はレンジ重複。TMA は `tma_survey.md` §1.4 の
  「DRAM 飽和で素のロードと 1.8% 以内」がこのカーネルの天井を既に押さえる。

- **p=63 `GEMM_FUSED` の L2 persist 窓（2026-08-30、`p63_gap_study.md` §28）**:
  `q` に persist + streaming miss を掛けると device **+55%**（571 → 886 µs）。
  属性が後続 GEMM まで残り L2 を捨てる。コードは戻した。

- **p=63 `GEMM_FUSED` の flux DRAM 取り方（2026-08-30、`p63_gap_study.md` §27）**:
  2 点 grid-stride は **+0.38%**、`__ldcs` は **+0.48%**、`__stcs` と L2 prefetch は
  レンジ重複。加重 z は ncu でも **254 レジスタ / 占有 12%**。コードは戻した。
  DRAM 83%→100%（~22 µs）は persistent grid-stride など未測の取り方を残す。

- **p=63 `GEMM_FUSED` の flux_yz 重ね（2026-08-30、`p63_gap_study.md` §26）**:
  `flux_y/z` を side2 で cuBLAS x と重ねると device **+5.9%**（571 → 605 µs/stage）。
  フルグリッドの DRAM カーネルが SM を占有して GEMM を直列化する。コードは戻した。

- **p=63 `GEMM_FUSED` の `Ey*acc+Ex*Dx`（2026-08-29、`p63_gap_study.md` §25）**:
  y エピローグで Ex も掛けて加重 z に渡し、scale カーネルと `GemmZWide` は使わない。
  占有 GPU 12 回交互で device **−4.1%**（595.2 → 570.9 µs/stage、max abs 3.55e-15）。
  nsys: z −64 µs、y +41 µs。p=31 では +2.3% なので `Nq==64` だけ。最速は
  `FUSED_TC` のまま。上表の 634.6 µs は §22 の別セッションで、書き換えない。

- **p=63 `GEMM_FUSED` の x/y 重ね（2026-08-29、`p63_gap_study.md` §24）**:
  y GEMM を side2 で cuBLAS x と重ね、占有 GPU 12 回交互で device **−0.28%**
  （596.2 → 594.5 µs/stage、ビット一致、レンジ非重複）。両方 SM ~68% なので
  91 µs の y はほぼ隠れない。加重 z への移植は重ねと両立せず **+2.9%**。
  屋根には当たっていない（z SM 38% / DRAM 36%）。測った候補は閉じたが、
  scale 無しの `Ey*acc + Ex*Dx` は未測。最速は `FUSED_TC` のまま。
  上表の 634.6 µs は §22 の別セッションで、書き換えない。
- **p=31 `CUDAFORTRAN_FUSED`（2026-08-30、`p31_gap_study.md` §20）**: CUDA-core
  融合を **992.5 → 718.2 µs/stage（−27.6%、ビット一致）**にした。job `69623`
  （c390）12 回交互 A/B。律速は長さ 32 の shared 内積の MIO。効いたのは
  512 スレッド `double2` + 要素 2 スラブ、xz 次平面の先読み、パネル二重バッファ。
  最速は `FUSED_TC`（359.7 µs）のまま。論文の主比は 2.78× → **2.00×**。
  内積を消した天井は −56% で、契約内の CC スケジュールでは探索終了。
  scale 無しの折り込みは §25。最速は `FUSED_TC` のまま。

- **計測区間（2026-08-27）**: `WarmupStep`（namelist、既定 1）で指定した先頭
  ステップは実行はするが計時に含めない。CUDA graph 経路では step 1 が直接
  ローンチ、step 2 がキャプチャなので自動的に 2 以上に引き上げられる。
  総ステップ数は `nstep` のままなので場の値は変わらない。報告に
  `Measured steps:` と `Main per step:` が加わる。**これ以前の表の数値は 1
  ステップ目を含んだ値**であり、短い `nstep` の測定（p=63/p=255 の
  `nstep=20`）では新しい測定の方が数 % 速く出る。表はそのまま残す。

- **Ozaki Scheme II（arXiv:2504.08009v3、2026-08-27）**: INT8 Tensor Core による
  FP64 GEMM エミュレーションは **本番最速経路としては不採用**（下記 `ozaki2_survey`）。
  `feature/ozaki` ブランチでは **`CUDAFORTRAN_GEMM_OZAKI2`** として volume GEMM
  置換を実装済み（`ozaki2_implementation_report.md`）。GEMMul8 参照実装と比較し
  moduli テーブルを整合。p=255 では native GEMM 比 **約 2.25 倍遅い**。
- **Ozaki Scheme I（2026-08-28）**: 同 worktree で **`CUDAFORTRAN_GEMM_OZAKI1`**
  を追加（`ozaki1_implementation_report.md`）。A/B 両方をスライスし最大 s² 本の
  INT8 GEMM を FP64 で直接加算（CRT なし）。p=7 では OZAKI2 より遅い典型（s=8 で
  device 約 3.2× native）。数値は OZAKI2 と同オーダーの量子化誤差。

- **cuBLAS FP64 emulation（2026-08-27）**: `CublasEmulation=.true.` は比較実験の
  ため `EAGER` を明示的に強制する。既定の仮数幅は **FIXED 55 bit**。
  `EmulationMantissaControl=DYNAMIC` で当時の ADP に戻せる。
  p=7 では native FP64 の **約131倍遅い**ため、
  計測は `nstep=1--10` に制限する。8 GiB workspace は初期化時に一度だけ確保して
  再利用するが、cuBLAS 13.2.1 の内部 `cudaMallocAsync` は残り、p=7 の不利は
  解消しない。詳細は `cublas_emulation_survey.md`。
  **（追記 2026-08-29）p=1023 でも EAGER+ADP は volume GEMM 3 本で 2.27× 遅い。**
  INT8 算術は 64 倍足り、FIXED 55 bit の x は 0.495× まで勝つ。負けは
  ADP の入力解析と、y の 1024³ batched が INT8 タイルを埋めないこと。
  同日、本体の既定を **FIXED 55** にし、namelist で DYNAMIC に切り替えられる
  ようにした（`cublas_emulation_survey.md` §6–§7）。
  本体の FIXED 55 vs native（3 run 中央値）は §8: p=511 / p=767 は勝ち
  （0.95× / 0.91×）、p=575 は 1.31×、p=7 は 305×。p=1023 native は
  193.1 ms/stage、FIXED 55 は 8 GiB workspace 先行確保で場コピー時 OOM。

- **p=7, `Ne=32^3`**: `CUDAFORTRAN_FUSED_TC` が最速（commit `e22dda1` 以降）。
  現時点の device 時間は 0.806 秒 / Main 1.066 秒（下の ±x 面の項目）。
  それ以前は `CUDAFORTRAN_FUSED` が最速だった。
  さらに occupancy を 100% に上げる作業（`tc_paper_survey_2407.09621.md` §7）で
  device 時間 1.076 → 0.851 秒。同じ知見を CUDA core 版にも適用すると
  そちらも 1.153 → 0.986 秒になり（§8）、両者を 100% occupancy で揃えた
  TC 版の優位は 1.16× である。
- **p=63 の `FUSED_TC` をさらに詰めた（2026-08-28、`p63_gap_study.md` §19）**:
  §18 の形から **507.1 → 487.8 µs/stage（−3.8%、`Main` 1.57204 → 1.51089 ms/step、
  §18 とビット一致）**。同一実行ファイルの `GEMM_FUSED` は 657.4 µs/stage なので、
  **p=63 の最速は `CUDAFORTRAN_FUSED_TC` のままで順位は動かない**。
  効いた 3 つはすべて「どの資源も飽和していない、動くのは占有率と
  mma が隠せない 1 本のロードだけ」という 1 つの診断から出ている。
  (1) **y カーネルだけを 2 ブロック/SM に**（−2.3%）。y は蓄積器が 1 組なので
  512 スレッドで 64 レジスタに収まり、xz は 2 組あるので 144 B スピルして +7.4% と負ける。
  **同じ 1024 スレッド/SM でも 2×1 タイルにして買うと +2.9% で負ける**ので、
  占有率はオペランドロード比を保てたときだけ払う。
  (2) **y エピローグの `dqdt` の read-modify-write を `cp.async` で shared に先読み**（−1.1%）。
  §16.6 が「レジスタが足りず塞がっている」と書いた道で、`cp.async` はレジスタを
  経由しない。`long scoreboard` が 2.54 → 0.34 になる。**副産物として
  §16.6 の「`sFV` 転置は 17% 遅い」が消えた**（差が無くなった）——
  あの診断が正しかったことの独立な確認である。
  (3) **`sFU` のストアをコンフリクトフリーに**（−0.7%）。`swt63` が `l` の下位 2 bit
  しか畳まないので**ストア側だけ 4-way**だった。`l` の bit 2-3 を bit 0-1 にも畳むと
  下位 4 bit 上で全単射になり、読みは `(idx>>8)&3` が `ks`（展開済みループの定数）
  なので壊れない。ストアのコンフリクト 3.495 M → 0.281 M。
  **`p63_gap_study.md` §17.1 の「p=63 の残るコンフリクトは構造的」は成立しない。**
  **測って落とした候補**: p=127 §11.10 の**スウィズル代数簡約は p=63 では +2.4%**
  （命令 −13% で遅くなる。占有率 24% の xz では、消した整数演算が mma 依存の隙間を
  埋めていた）、staging ループの統合（+1.2%、転送は減るのに依存鎖が伸びる）、
  xz エピローグの `cp.async` 先読み（`Escale` +6.9%、`flux` +4.7%、両方 +17.1%）、
  1 ブロックに j 平面 2 枚（+4.8%）、xz を x と z に割る（y 170 µs × 2 > 融合 xz 317 µs で
  **割る前から負け**、実装せず）。**アブレーションで測った残りの天井は
  mma 全消しの 38.7 µs（7.8%）が最大**である。
  そして **§19.7 は ncu と実時間が系統的に食い違う例を 3 つ記録している**:
  ncu で y カーネルが −8.3% になる `sFV` の `swu63` は、実時間では +1.0% で負ける。
  ncu はクロックを固定するのでレイテンシ律速のカーネルでは shared の節約が
  過大に見える。**採否は占有 GPU 上の交互 A/B で決めること。**
  **§19.10 で 5 例に増え、機構が特定できた**: SM クロックを固定で落とすと
  global レイテンシがサイクル数で短くなるので、**ncu は shared・レジスタ・命令の
  仕事を高く、global の待ちを安く見積もる**。shared を減らす 4 件は ncu が
  過大評価し（−2.4〜−8.3% に対し実時間は ±0〜−0.7%）、**shared を増やして
  global の待ちを消す p=31 の先読みは ncu が +8.3% と誤って棄却させる**
  （実時間は −0.51%）。**この 2 つを取引する変更は ncu で採否を決めてはならない。**
- **p=63 の残り 3 ブロックの天井を測って探索を終了した（2026-08-28、
  `p63_gap_study.md` §20）**: 採用ゼロ。§19.8 の「残る候補はどれも天井が測ってある」は
  **xz カーネルについてだけ**の話だったので、y カーネル・xz の shared・
  面フラックスカーネルの 3 つを埋めた。
  (1) **y カーネルは識別できる仕事を全部消しても 32.4 µs**（`Escale` 9.1、mma 7.8、
  shared 7.4、`dqdt` 6.4、staging 1.7）で、単独最大が 1.8%。
  (2) **xz の shared トラフィックは消すと +0.93% 遅くなる**。§19.8 が
  「残り 180 µs は shared トラフィックとバリアとレイテンシ」と推定していたのは
  誤りで、**xz の床はワープ並列度そのもの**である（p=127 §11.12 と同じ）。
  (3) **面フラックスカーネル（`elembnd_flux_kernel`）が実は 1 stage の 13.45%（67.1 µs）
  あり、その 79%（53.3 µs）がギャザーの非効率だった。** ncu は 10% と出していた ——
  §19.10 のモデルどおりの過小評価である。内訳は速度 `u`/`v`/`w` の 6 本が 35.0 µs、
  `q` の 2 本が 13.4 µs。**原因は Nq=64 のノード番号 `i + Nq*j + Nq²*k` で
  面 2/4（i 一定）に連続方向が存在しないこと**で、1 レーンが 32 B セクタから 8 B しか
  使わない。**スレッド割り当てでは直らない**: 1 スレッド複数面点（`FLUX_PTS`=2/4/8）は
  +0.5〜+2.3%（占有率を食うだけで MLP は足りている）、`intent(in)` による
  read-only キャッシュ経路（SASS で全ロードが `LDG.E.64.CONSTANT` になることを確認）は
  ±0。**唯一残る手は `VelM`/`VelP` を面点ごとに 1 度だけ計算して保存すること
  （天井 35.0 µs = 7.0%）だが、これは「呼び出し間で速度が変わらない」という
  呼び出し側への仮定で、`AGENTS.md` の配列インタフェース要件に触れるため範囲外。
  **この事前計算は行わないと判断済みで、35.0 µs は実装しないことのコストとして
  受け入れる（`p63_gap_study.md` §20.4）。再検討しないこと。**
  **契約の中で実装できる候補は残っておらず、p=63 `FUSED_TC` の探索を終了した。**
- **p=63 `CUDAFORTRAN_FUSED`（CC）を 966.4 → 776.0 µs/stage にした（2026-08-30、
  `p63_gap_study.md` §24）**: job `69633` 交互 12 回、−19.7%。効いたのは
  `__restrict__`（−6.4%）とチャンクループ除去（平面 64×64 を動的 shared）。
  xz は L1/TEX 85% の MIO 律速で、チャンクを消すと 94% まで張り付く。
  計算 LDS を消すアブレーションは −58% だが、`double4` / shuffle / `cp.async` /
  unroll は全部負け。最速は `FUSED_TC`（421.7）のまま。主比 2.29× → **1.84×**。
  **（訂正・同日、§25）** 「CC 側の探索は終了」は早かった。512 スレッド化で
  さらに 776.0 → **618.6 µs/stage（−20.3%）**、主比 **1.47×**。§24.5 の表は
  1024 スレッド時点の負例として残す。

  **（訂正・同日、§20.6）** 「TMA でまとめて持ってきて端だけ使えばよいのでは」という
  指摘を受けて前提を測り直し、**§20.3 の 53.3 µs が過大だったこと**を訂正した。
  線形化アブレーションは増幅を消すと同時に**触るデータ量そのものを小さくする**
  （ncu 実測で DRAM 読みが 372.3 → 113.3 MB、後者はこのカーネルが触る distinct な
  データ量そのもの）。**帯域は余っておらず、現行は DRAM の 76.1%** で 1 stage 中
  最も帯域に張り付いている。**TMA でも減らない**: 面 2/4 が要る i 平面に対し
  32 B セクタ粒度が下限を決めていて、現行のギャザーは既にその下限（1 面 1 フィールド
  128 KB）を払っており、box を広げれば単調に損（要素まるごとで 16 倍）。
  **shared に載せて効く再利用も無い**: 周期境界で境界ノードは M/P の 2 回読まれるが、
  **P 側を完全に無料化しても −4.7 µs（0.95%）**しかなく、面点ペアの対応表を
  `mod_mesh` に持たせる設計は天井が実装コストに見合わない。
- **横展開（2026-08-28、`p63_gap_study.md` §19.9）**: **p=31 の `FUSED_TC` に
  `dqdt` の `cp.async` 先読みを移して −0.51%（ビット一致、レンジ非重複、
  440.9 → 438.8 µs/stage）**。p=31 の y カーネルは 16 枚の k 平面を順に回るので、
  平面 `kl` の mma 中に平面 `kl+1` を積む 2 段にした（`long scoreboard`
  10.38 → 1.39）。`q`/`v` には既に同じパイプラインがあり、`dqdt` だけが
  取り残されていた。**p=127 では入らない**（1 ブロックのタイルが 64 KB で
  shared が 227 KB を越える）。
  **`p127_gap_study.md` §11.13 が棚卸ししていた `sFU` のストアコンフリクトは
  取り消せるが、取り消しても時間にならない**: `swu128` でストアコンフリクト
  7.84 M → 2.54 M、ncu では xz −2.4%、**実時間は完全な引き分け**なので戻した。
- **p=63, `Ne=4³`（2026-08-28、`p63_gap_study.md` §16）**: **最速は `CUDAFORTRAN_FUSED_TC`**
  （**533.7** 対 `GEMM_FUSED` 587.6 µs/stage、**1.101 倍**）。前日の 662.3 µs から **−19.3%**。
  **（追記・同日、§18）その後 477.2 µs/stage になった**（`GEMM_FUSED` 573.0 に **1.201 倍**）。
  チャンクループ本体の末尾に置いていた `__syncthreads` を先頭へ `if (kk)` 付きで移すと、
  `BK63 = NQ63` で 1 回しか回らないこのループでは末尾の 1 本がまるごと消える。
  **ビット一致で −7.7%。** 最速経路は変わらない。以下は当時の記述である。
  効いたのは 2 つで、どちらもレイテンシに効く変更である。
  (1) **チャンクループを消した**（`BK63` を 16 → 64、パネル 3 枚 96 KB の動的 shared）。
  プリフェッチが無いループは 1 チャンクごとに止まるが、`BK=Nq` ならループ自体が無く
  平面の global ロードが全部 mma の前に飛ぶ。占有率はレジスタで決まるので shared 増は無料。
  (2) **`sD` と `sFW` を転置レイアウトにした**。
  **副産物として `AGENTS.md` に効く一般則が確定した**: 8 B の shared アクセスで
  コンフリクトフリーの条件は「32 レーンで `d mod 32` が相異」ではなく
  **「半ワープ 16 レーンで `d mod 16` が相異」**である（§16.4、SASS で `LDS.64` 128 本が
  両版同一であることを確認したうえで実測と一致）。この誤りのせいで一度は
  「両側コンフリクトフリーは原理的に無理」と結論しかけた。
  (3) mma が読むパネルは `sFV` を除き全部 outer-fast にする（§16.8）。
  **`sFV` の逆転は解決した**（§16.6）: `sFV` の転置は y カーネルを **7% 速くする**のだが、
  短くなった mma がエピローグの `dqdt` read-modify-write を隠せなくなり、
  `long scoreboard` が 31% → 50% に跳ねて差し引き 22% 遅くなる。
  `dqdt` の読みを消すアブレーションで転置の代償が +16.8% → **−0.3%** になることで確定。
  read-modify-write を mma の長い xz 側へ移す修正も試したが、コストが移るだけだった。
- **バンク判定を全次数に当て直した（2026-08-28、`p63_gap_study.md` §17）**: 採用ゼロ。
  **p=7 のカーネルは最初から正しい判定で書かれており**（冒頭コメントに half-warp / `d mod 16`
  と明記）、§16.4 の誤りは既存知識の退行だった。残るコンフリクトはどれも構造的で、
  p=7 は蓄積器対の `double2` 隣接（自由なビットが 3 本しかなく 2-way が下限）、
  p=63 は global と mma のレイアウトが直交していること。
  p=31 / p=15 / p=255 はコンフリクトが 0.7〜7.0% と小さく律速も別（`lg_throttle` や MIO）。
  **唯一 p=7 だけは条件が揃っている**（占有率 96.3%、L1/TEX 91.2%、`long_scoreboard` 50.2%）
  が、**三度目も失敗した**（§17.3、+3.81%）。失敗の仕方が理由を教えた:
  **ptxas は隣接する 8 B ストア 2 本を `STS.128` にマージする**ので、
  **bit 0 への畳み込みはストアからは見えず**読み側だけを壊す（st conflicts は 8% しか減らず、
  ld conflicts が 4.6 倍）。16 B 単位で数え直すと**蓄積器のストアは現行 `sw_dz` で既に
  コンフリクトフリー**である。残る箇所をアブレーションで帰属させると**支配的な出所は無く**、
  **最大の塊（flux パネル、コンフリクト −21%・shared wavefront −8.4%）を消しても −0.2%**（§17.4）。
  **L1/TEX 91% は律速ではなく**、p=7 は `long_scoreboard` 50%（global ロード待ち）で決まる。
  過去 2 回 + 今回 1 回、計 3 回の失敗はすべて「賞金が 0.2% 以下だった」で説明がつく。**探索終了。**
- **p=63 の旧記録（2026-08-27、`p63_gap_study.md` §13）**: 実装当初は
  **`GEMM_FUSED` が最速**だった（598.6 対 `FUSED_TC` 662.3 µs/stage）。
  Nq=64 は p=31 の設計の前提を 2 つ壊す（`Ne=4³` しかなく 1 要素 1 ブロックでは 152 SM が
  埋まらない、1 平面 32 KB で x と z のパネルが両方載らない）ので、面 flux を
  `elembnd_flux_bnd` に先出しし、縮約を `BK=16` でチャンク化して 3 枚 24 KB に収めた。
  swizzle は **p=255 の `sw255` をそのまま流用**でき（レイアウトが同じ `l + 16*outer`）、
  ロードのバンクコンフリクトは 0 件。**p=31 の勝因だった D1D のレジスタ常駐は
  Nq=64 では成立しない**（1 レーン 64 double 要る）。mma は shared wavefront を
  5 分の 1 にしたのに時間は 1.6 分の 1 にしかならず、**その先で何も律速になっていない**
  （SM 39%、DRAM 18%、L1/TEX 53%、占有率 24.6% のレイテンシ律速）。
  **これで「融合が GEMM に勝つ上限は p=31」が確定した。**
  warp 形状は 3 通り全部測り、p=255 と同じ 2×4 は 198 レジスタ・占有率 12.5% で最悪、
  2×2（512 スレッド）が最良、2×1（1024 スレッド、占有率 50%）は得ゼロだった。
  詳細は `p63_gap_study.md` §13。
- **p=15, `Ne=16³`（2026-08-28）**: `CUDAFORTRAN_FUSED_TC` を **345.1 → 326.8 µs/stage
  （−5.3%、ビット一致）**にした。面フラックスに専用 shared バッファ 12 KB を与え、
  z フェーズの後ではなく **x パネル格納の直後**で評価する。`sbuf` を体積パネルと
  共用していたことが評価位置を縛り、面 gather を「誰もが最初に待つもの」に固定していた。
  **global のリクエストもセクタも 1 つも減っていない**（4.63 M / 62.36 M で同一）。
  減ったのは `long scoreboard` だけで 47.1% → 24.5% とちょうど半分になる。
  占有率はレジスタ（64 × 1024 = レジスタファイル全体）で決まるので 12 KB は事実上タダ。
  1 段早く（x パネル格納の前に）置くと −4.4% と悪くなり、専用バッファを持たず
  レジスタに退避すると spill で **+2.0%** と逆効果。詳細は `p15_gap_study.md` §15。
  **（追記 2026-08-28）この節の「12 KB は事実上タダ」は正しかったが、
  そこで止めたのは早かった。`p15_gap_study.md` §16 は同じ論法を最後まで押して
  shared を 47872 → 129792 B に増やし、272.4 µs/stage まで下げている。**
- **p=15 `FUSED_TC` の一枚バッファ設計をやめた（2026-08-28）**: **331.2 → 272.4 µs/stage
  （−17.8%、全段ビット一致。Main は 1.2375 → 1.0678 ms/step、CUDA graph on で
  1.2163 → 1.0376 ms/step）**。§15.7 が「残り 45 µs、取る手が無い」で終わっていたところを、
  **`shared` を節約するという前提そのものを捨てて** 5 つの変更で 60 µs 取った。
  最初に測ったのは**カーブアウトの代金**で、占有率がレジスタ（64 × 1024）で
  1 ブロック/SM に固定されているため **+64 KB までは実質タダ（+1.4%）、+128 KB で +9.4%
  の崖**がある。これに乗せて (1) 面フェーズを 2 面点/スレッドにして `normal_fn`/`Fscale`/
  `VMap` を `double2`・`int2` 化（−3.0%、request −12.7% だが**セクタは +13.7%** という取引）、
  (2) **x/y/z の 3 パネルを同時に shared に置く**（+64 KB で `__syncthreads` が 8 → 3 本、
  バンクコンフリクトが 0 に、−7.0%）、(3) `__restrict__`（−2.2%）、
  (4) **M 側の i 境界 2 面（16 KB）を体積フェーズの通りがかりに shared へ書き、
  面 gather をそこから読む**（−6.1%。§14.6 が残した唯一の筋で、128 B 飛びの
  ワープが 32 セクタ引いていた 2 面が消える。`VMapM` が返したノード番号で判定するので
  マップは仮定していない）、(5) z 往復の上書きを**そのワープ自身が読んだ番地**に
  落として `__syncthreads` を `__syncwarp` に（−0.8%）。
  **(1) と (2) は足し算より大きい**（単独 −3.2% / −3.7%、両方で −9.8%）。
  **`__restrict__` は次数をまたぐ**: p=31 −4.0%、**p=63 −11.6%**、p=127 −1.2%、
  p=255 `FUSED_TC` −5.7%。**p=7 だけは +3.6% と逆効果**で、32 レジスタ 8 ブロック/SM の
  設計が 8 バイトの spill を出すため付けない。**測って落とした候補**: 面 gather 自体の
  `double2` 化（±0 —— 命令もセクタも減るのに動かない）、`ld.global.cs`（+15%）、
  `VMap` の前倒し（+5.5%）、z を先に回す並べ替え（+3.2%）、D フラグメントの
  レジスタ常駐（+0.5%）、6 面全部の M 側常駐（上限 −3.8% に対し carveout +2.6% で
  書く前に赤字）。**残りの床は 223.0 µs**（両側 coalesced のアブレーション、−18.0%）で、
  §15.7 の 281.3 から 58 µs 下がった。詳細は `p15_gap_study.md` §16。
- **p=255 / p=127 の `GEMM_FUSED` を詰めた（2026-08-28、`p255_gap_study.md` §10、
  `p127_gap_study.md` §12）**: z の assembly epilogue が**命令発行律速**である
  ことを ncu で確定させ（lift を消すと命令数 −13.0% で時間 −12.8%、stall 内訳も
  占有率も動かない。レジスタは 254 → 184 に落ちるが、ブロック数はレジスタと
  shared 48 KB の**両方**で 4 に制限されているので占有率は変わらない）、
  命令数を減らす変更を 3 つ入れた。いずれも `Nq > 64` の枝のみ。
  (1) **`Escale_x` / `Escale_y` の乗算を x / y GEMM の epilogue へ前送り**（−1.0%）。
  z が読む体積テンソルは 5 本から 3 本になる。転送量は同じで、**402 MB が
  SM 73% / DRAM 20% の z から、SM 88〜90% / DRAM 6〜10% の x/y へ移る**。
  要点は**手書き epilogue を書かないこと**で、CUTLASS 標準 epilogue の source
  経路に載る出力オペレータ（`D = acc * C`）として書く。手書き epilogue を
  y GEMM に入れる版は**それだけで +72 µs**かかり、移動で得られる分の 4 倍を失う。
  (2) **添字クランプをタイル原点へ集約**（−0.85%）。はみ出した行は「本来とは違うが
  有効な」アドレスを読むだけで、値は出力イテレータが述語化するので書かれない。
  これで面 gather のアドレスが行オフセットについてアフィンになり強度低減が効く。
  (3) **lift の 6 本のロードを 3 本の 16 バイトロードに**（−0.75%）。面 2/4、面 1/3、
  `Lift1D` の面 5/6 係数はどれも添字を共有するので、`elembnd_flux_kernel` に
  面平面をペアでインターリーブさせ、`Lift1D` の z 面 2 列は初回に 1 度だけ詰める。
  **p=255 は 1048.8 → 1021.2 µs/stage（−2.6%）**で `FUSED_TC` 1015.0 との差は
  1.035 倍から **1.006 倍**に縮んだ（最速は `FUSED_TC` のまま）。
  **p=127 は 789.8 → 752.8 µs/stage（−4.7%）で最速が `GEMM_FUSED` に戻った**
  （`FUSED_TC` 787.8 に 1.047 倍）。次数が低いほど効きが大きいのは、epilogue の
  仕事が出力点数に比例する一方 mma は `Nq^4` に比例するためである。
  p=63（`Nq<=64`、x GEMM が cuBLAS）は**ビット一致で時間も不変**。
  (2)(3) を p=63 の枝にも当てると、ビット一致のまま **+2.7% / +0.8%** と遅くなる。
  不採用の記録: z を 2 テンソル化（+8.1%）、z のタイル・warp・stage・padding 掃引、
  x GEMM の batched 化（+0.7%）・cuBLAS 化（+0.7%）・タイル掃引、
  epilogue への `__restrict__`（±0）。**§10.8 で残りも閉じた**: lift の shared
  ステージは命令発行律速なので賞金ゼロ、lift を x/y GEMM へ移す案は
  1 要素あたりのロードが z 3 本・y 3 本・x 5 本で z が最良の宿主、K 拡張で
  GEMM 本体に載せる手は**その GEMM の epilogue が `Escale` 乗算に使えなくなる**
  ため 4 通りすべてでロード数が変わらず差し引き +3.4 µs、エポローグ全展開 +1.0%、
  占有率を上げる `__launch_bounds__` は 4/5 ブロックともスピルして +24〜40%。
  **残るのは 3 本の GEMM の mainloop（ピーク比 84〜86%）だけで、同じ形状で
  cuBLAS も同じ 85% しか出ない。**

- **`GEMM_FUSED` の z を 4 ストリームにした（2026-08-28、`p127_gap_study.md` §13）**:
  **p=127 は 752.4 → 731.7 µs/stage（−2.8%、`Main` 2.2883 → 2.2231 ms/step）、
  p=255 は 1020.6 → 1004.5（−1.6%）、p=63 はビット一致で不変。**
  p=127 の最速は `CUDAFORTRAN_GEMM_FUSED` のままで、`FUSED_TC`（同日 781.0）
  との差は 1.047 → **1.069 倍**に広がった。p=255 の最速は `FUSED_TC`
  （同日 961.0）のままである。
  上の §10 は z の epilogue を**命令発行律速**として詰めたが、そこから先の
  z はメモリ側で詰まっている。アブレーションで天井を出すと
  **134 MB 消しても 403 MB 消しても −72 µs で止まる** —— 帯域なら線形に効くはずで、
  実際には z が **mainloop 床 139 µs**（mma 下限 107.1 µs の 1.30 倍、
  tensor pipe 77%、x/y の 81〜84% と同水準）に当たって止まっている。
  効いたのは 3 つ。
  (1) **`deriv_x` を y GEMM の 2 ソース epilogue（`deriv_xy = Escale_y*acc + deriv_x`）
  に畳む**。z の読みが 4 本 → 3 本になり **z −15.7 µs に対し y は +2.1 µs しか
  払わない**。y は DMMA パイプ 81% / DRAM 18% で余裕があり、しかも
  **`deriv_x` は直前の x GEMM が書いたばかりで L2 に温かい**（z が読むときは
  あいだに y が 400 MB 流したあとで冷えている）。GEMM チェーンの転送は
  6 本の読み + 2 組の中間 + `dqdt` の書き = 11 単位で構造的に固定なので、
  **動かせるのは総量ではなく「どのカーネルが払うか」だけ**である。
  §10 の「手書き epilogue は +72 µs」は**標準 epilogue を置き換えた場合**の話で、
  z と同じく `Epilogue` から派生して `OutputTileIterator` を 1 本足すだけなら
  その代償は出ない（`cutlass_y_gemm_scaleadd.h`）。
  (2) **lift の積和をアキュムレータの shared 往復の後ろへ回す**（ビット一致、−5 µs）。
  面 2 本のロードだけ先に出し、バリア 2 本と `acc2smem` でレイテンシを覆う。
  (3) **epilogue のアクセスを 16 バイトに**（p=127 −7.2、p=255 −17 µs）。
  **32 バイトは赤字**（z +35、x/y +7 µs）で、**`Nq<=64` では 16 バイトも +2.8%**
  なので `GemmZWide` を別型に立てて `kWeighted` で選ぶ。
  数値は HEAD とビット一致ではない（y の `ad + acc*sc` が FMA に縮約される）が、
  和の順序は同じで、`OPENACC_SPLIT` に対する最大絶対差は
  **3.553e-15 → 1.776e-15 と改善**、>1e-14 は 0 件。
  **落とした候補**: CTA のワープ数を増やす（x `<32,32>` 767.2 / `<16,64>` 783.1、
  y 782.4 / 773.5、z 789.2 / 830.5、基準 752.5）—— 1 スケジューラ 2 ワープでも
  IPC は 0.24/1.0 で**発行スロットは元から余っている**ので占有率は効かない。
  z のタイル `<64,64>`（757.3）とスウィズルの n 優先グループ化（差なし）——
  `flux_z` の 4 回読みは **ncu の `dram__bytes` で既に 1 回分**しか出ておらず
  L2 が捕まえている。epilogue ループの 2 段 / 全展開（761.8 / 755.7）。
  **計測衛生（§13.8）**: `it_dy` の `++` を 2 行動かすだけで、ビット一致のまま
  **+3.5%**（752 → 778 µs/stage）になる。この epilogue の A/B は
  「マクロを足しただけ」の版でも必ず基準を測り直すこと。
  **§13.10 で最後の筋も閉じた**: lift を y GEMM の epilogue へ移すと、
  索引が良くなって 1 要素あたりの面ロードが 2 本 → 1 本になり、
  z は予測どおり **−30.0 µs** 手放すのに、**y が +43.9 µs 払って差し引き +15.0 µs**
  になる。**「どのカーネルが払うか」の論法は転送には効いて命令には効かない** ——
  転送は z が余らせていない資源で y が余らせている資源だったが、
  命令はどちらも同じ発行資源で、y のほうが（DMMA パイプ 81%、余裕 27 µs）
  先に埋まっている。`lz` だけ z に残す版はさらに悪い（+8 µs、per-row のループ足場が
  節約した 2 FMA より高い）。副作用として、y が `flux_bnd` を読むと side stream の
  join を x と y のあいだに入れる必要があり**それだけで +6 µs**。
  **残る筋は tcgen05 だけで、FP64 のピークは変わらないので天井は上がらない。**
  改修後の ncu（job `66552`）はモデルの答え合わせになっている: **z は命令 −10.9% で
  SM が 63.2 → 70.1%**（メモリ側の露出が減って演算の屋根に寄った）、
  y は `deriv_x` を引き受けて DRAM 17.9 → 24.5% でも SM は 78.7 → 76.9% とほぼ不動、
  x は 16 バイト epilogue で命令 −6.2%。
  **プロファイラ側の落とし穴（§13.8）**: 同じ `ncu --set full` が **ノード c391 では
  40 分無出力で `timeout`**、c160 / c070 / c398 では 42〜43 秒で完走する。
  返ってこないときは `sacct --format=NodeList` を見て `#SBATCH --exclude=<node>`。

- **p=255, `Ne=1`（2026-08-28、`p255_gap_study.md`）**: **最速は `CUDAFORTRAN_FUSED_TC`**
  （Main **3.1265** 対 `GEMM_FUSED` 3.2346 ms/step、**1.035 倍**。
  両者が出す `Volume derivate + surface lift` で 1014.2 対 1050.5 µs/stage）。
  改修前の `FUSED_TC` は 1563.9 µs/stage だったので **−38.1%**、
  全段が改修前と**ビット一致**である。効いたのは 4 つ:
  (1) チャンクループの**二重バッファ化**（`issue(k+1) -> mma -> store -> barrier`、
  バリアは 2 本から 1 本）と **1 ワープ 4×4 mma タイル / 128 スレッド**。
  この 2 つは**組でしか効かない**——4×4 単独では蓄積器 64 レジスタで占有率が
  半減して 23.9% 遅く、二重バッファ単独では 14.8% しか速くならないが、
  組にすると **−27.7%**。`__launch_bounds__` の第 2 引数は当たりくじではなく
  **スピルしない最大の占有率**（168 レジスタ × 128 スレッド × 3 ブロック =
  レジスタファイルのほぼ全部）である。
  (2) staging の **double2 化**（−3.7%）。
  (3) mma ループの**スウィズル・ハイスト**（−1.7%）。`sw255` は index の下位
  4 bit しか触らないので、オペランドごとの項と k ステップごとの項に分かれる。
  (4) **エピローグを b 外 / a 内に組み替える**（−8.6%）。x では面フラックスが
  m だけ、lift 係数が n だけで決まるのに平坦に回していたため、1 ワープ
  16 出力ペアに **112 本の LDG** が出ていた（→ 32 本）。
  さらにストアのペア格納順を半ワープの上位 8 レーンだけ入れ替えると
  2-way バンクコンフリクトが消える（8.45 M → 70 K、−0.8%）。
  **手書き GEMM が CUTLASS `d884gemm_64x128_16x3` を上回れた理由は演算ではなく、
  エピローグと面 lift を同じカーネルで払えることである。**
  **命令数は律速ではない**という証拠が 2 つ出た: 単一バッファ時の
  スウィズル・ハイストは命令 −17% で **+5.1%**（パイプライン後は逆に −1.7%）、
  staging アドレスのハイストは命令 −34% で **+2.5%**。
  最終形の stall は `math_pipe_throttle` 3.21 + `wait` 2.90 が 3 分の 2 を占めるが、
  FLOP を 81% 落とすアブレーションは 2.8% しか速くしない——
  **どの資源も単独では律速していない**。
- **p=511, `Ne=1`（2026-08-28、`p511_gap_study.md`）**:
  `CUDAFORTRAN_GEMM_FUSED` が最速。warm-up 後の device tendency は
  **13.159 ms/stage**、`CUDAFORTRAN_GEMM` の 13.528 ms/stage より **2.73%**短い。
  点変化係数を含む全134,217,728点の `dqdt` を最大絶対差3.55e-15で検証した。
- **p=575, `Ne=1`（2026-08-28、`p575_gap_study.md`）**:
  `CUDAFORTRAN_GEMM_FUSED` が最速。3-run中央値の warm-up 後 device tendency は
  **20.453 ms/stage**、`CUDAFORTRAN_GEMM` の 20.912 ms/stage より **2.20%**短い。
  点変化係数を含む全191,102,976点の `dqdt` を最大絶対差3.55e-15で検証した。
- **p=767, `Ne=1`（2026-08-28、`p767_gap_study.md`）**:
  `CUDAFORTRAN_GEMM_FUSED` が最速。3-run中央値の warm-up 後 device tendency は
  **60.362 ms/stage**、`CUDAFORTRAN_GEMM` の 62.817 ms/stage より **3.91%**短い。
  点変化係数を含む全452,984,832点の `dqdt` を最大絶対差3.55e-15で検証した。
- **p=1023, `Ne=1`（2026-08-28、`p1023_gap_study.md`）**:
  `CUDAFORTRAN_GEMM_FUSED`が最速。3-run中央値のwarm-up後device tendencyは
  **187.617 ms/stage**、`CUDAFORTRAN_GEMM`の194.058 ms/stageより**3.32%**短い。
  両経路のrequested payloadは144.320 GiBだが、OpenACC allocator込みの実測peak
  増分は176.4 GiB。p=511/575/767/1023を覆う保守見積もりは`payload*1.25+2 GiB`。
  点変化係数を含む全1,073,741,824点の`dqdt`を最大絶対差3.55e-15で検証した。
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
  **（訂正 2026-08-28）「問題にならなかった」のは 96 KB を避けたからだが、
  避ける必要は無かった。`p15_gap_study.md` §16.2 が carveout の代金を測ると
  +64 KB までは +1.4%、実際に 3 パネルを同時に置くと −7.0% である。**
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
  交絡（レジスタ絞りで +6.8%、L1 半減で +0.4%）を分離すると、
  **占有率上昇それ自体の価値は約 −0.8%**。
  **「レイテンシ律速なら warp を増やせば隠れる」が成り立たない理由**は ncu の
  warp 統計に出ている: warp/scheduler を **1.96 倍**にすると
  1 命令あたり stall サイクルが **1.86 倍**に伸び、発行率は **1.05 倍**しか動かない
  （0.22 → 0.23）。発行率 = warp 数 / 命令あたりサイクル数がほぼ保存される、
  すなわち **`long scoreboard` は隠すべき固定レイテンシではなく、
  飽和しかけた L1/TEX の待ち行列遅延**であり、並列度を上げても
  サービス率は変わらず行列が伸びるだけである（Little の法則）。
  baseline の時点で L1/TEX 経路の余地は 1/0.759 = 1.32 倍しかなく、
  それを取りに行くレジスタ絞りが単独で +6.8% なので最初から赤字である。
  L1/TEX ヒット率は 2.8% しかなく、効いているのは L1 の再利用ではなく
  **L1 を経由する命令の本数**である。
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
  **体積ロードのベクトル化（2026-08-27）**: 「効いているのは L1/TEX を通る命令の本数」
  という診断から、ノードの持ち方を 1024 飛び 4 個から**隣接 2 ノード × 2 組**に変えて
  `q,u,v,w` を `double2` 1 本で運ぶようにした。どの swizzle もビット 0 を読まないので
  ペアは swizzle を通してもペアのまま。**global ld 命令 −18.5%、shared st 命令 −25.2%、
  一方で global ld セクタ数と DRAM バイト数は 1 つも動かない**まま
  device **21.75 → 20.86 ms（−4.1%）**、ビット一致。診断の直接の裏取りである。
  最終形は **347.7 µs/stage、5.98 TFLOP/s（14.9%）、3.91 TB/s（50%）**、
  CUDA core 版比 **1.249 倍**、TC 初版から通算 −6.2%。
  **この手は他パスにはほぼ使えない**: p=7 `FUSED_TC` に適用すると **1.8% 遅い**
  （DRAM 72.8% で帯域に張り付いており、命令を減らしてもバイトが減らないため。差し戻し済み）。
  CUDA Fortran のカーネルでは**そもそも書けない**（nvfortran 26.3 は 128 bit global ロードを
  出さない。実カーネルで `LDG.E.128` が 0 本、最小テストでも `complex(8)` でも同じ）。
  したがって前提が成立している p=15 `FUSED` にも適用できない。
  **比較の公平性**: 1.249 倍のうち 4.1% 分は CUDA core 側に持ち込めない最適化に由来し、
  両者に同じ最適化だけを許した時点の比は **1.196 倍**である。
  詳細は `p15_gap_study.md`。

- **p=63, `Ne=4³`（2026-08-27）**: 同一 DOF の 4 点目で、**GEMM 側の最初の点**。
  最速は **`CUDAFORTRAN_GEMM_FUSED`**、Main **0.04583** 秒 / **674.1 µs/stage**
  （graph on 0.04499）。本作業の前は `operator_data/p63.dat` が無く
  **どの実装でも起動できなかった**ので、`generate_lgl_operators_p255` を任意次数に
  一般化して p=31 / 63 / 127 を自動的に通るようにした（p=255 では旧ルーチンと
  **ビット一致**）。密な `Lift_mat` は分離形なので `Lift1D` から再構成する。
  併せて `mod_dg_optr_kernel.f90` の 4 か所の `select case(Nq-1)` に
  **`case default` が無く、p≥16 では `intent(out)` が未書き込みのまま
  エラーも出ずに返っていた**のを塞いだ。
  `GEMM_FUSED` / `GEMM_CUTE` の p=255 ゲートは C++ 側が元から Nq/Ne 汎用なので
  外せたが、**y GEMM の batch が `grid.z` に載るため `Nq*Ne <= 65535` が真の上限**で、
  p=7 `Ne=32³` と p=15 `Ne=16³` はこの経路を使えない（ゲートをその検査に置き換えた）。
  **Nq=64 では要素が shared に載らない**（1 要素 2 MB）ので融合カーネルは書いていない。
  書かない根拠は演算側にある: 融合の理想転送量 671 MB/stage は帯域下限 85 µs を与えるが、
  6.811 GFLOP を 674 µs 未満で回すには FP64 ピークの **25.2% 超**が要り、
  手書き融合がこれまで出した比は p=7 で 12.8%、p=15 で 14.9% にすぎない。
  **融合が勝つ p≤15 と GEMM が勝つ p≥63 の交差は p=15 と p=63 の間にある。**
  （**訂正 2026-08-27**: Tensor Core の融合は p=31 でも勝つので、交差は
  **p=31 と p=63 の間**である —— `p31_gap_study.md` §14.1。）
  達成効率は **10.10 TFLOP/s（25.2%）/ 3.96 TB/s（50%）** で、同一 DOF 4 点の
  FP64 ピーク比は **12.8 → 14.9 → 25.2 → 64.9%** と単調に上がる。
  **p=63 の tendency に単一の律速は無い**: `volume_flux`/`elembnd` は DRAM の屋根
  （85% / 76%）、x/y GEMM は SM 発行律速（74% / 67%）、最大の z GEMM+assembly
  （33.9%）はどちらでもないレイテンシ律速（SM 38% / DRAM 37%、レジスタ 254 本）。
  **GEMM 経路の転送量は Nq に依存しない**（159 B/ノード対 p=255 の 151 B）ことの
  直接の証拠は、`volume_flux_kernel` が両次数で **129.7 対 129.5 µs、
  実行命令数まで同一**なことである。`K=Nq` が浅い代償は命令効率に出ていて、
  x GEMM の FLOP/命令は p=255 の 135 に対し **48.5**。
  **事前予測は 2 つとも外れた**（「DRAM 側に寄る」「y GEMM が弱点」）。
  副産物として `cuda_cal_dqdt_gemm` の側ストリーム条件を `Nq == 256` から
  **`Nq >= 64`** に変えた（p=63 で Main −1.4%）。
  次にマシンバランスを越えるのは `6*Nq+20 = 5.0*151` から **Nq ≒ 123**、
  すなわち **p=127** のはずである。詳細は `p63_gap_study.md`。

- **p=63 の x GEMM を cuBLAS に渡した / `nstep=20` は短すぎた（2026-08-27、`p63_gap_study.md` §10）**:
  上の項目が「弱いのは x GEMM」と特定した点を追った。cuBLAS と CUTLASS は
  **同一のタイル・TileK・stage 数・命令形状**（`64x128_16x3`, m8n8k4）を選ぶのに、
  p=63 では cuBLAS 115.7 µs 対 CUTLASS 184.4 µs。ncu が理由を出していて、
  **同じ仕事を 21.2 M 対 44.3 M 命令**でやっている（spill は無い）。
  `K=Nq=64` が `TileK=16` で mainloop 4 反復しかなく、CUTLASS 2.x の
  prologue/epilogue が薄まらない。`Nq <= 64` のときだけ x を cuBLAS に渡すようにして
  tendency **660.4 → 590.0 µs/stage（−10.5%）**。出力は不変。
  `Nq=256` では **990.7 対 998.6 µs/stage で効かない**（mainloop が 16 反復ある）ので
  deep 側は CUTLASS のまま。これは H100 で `CutlassMmaShape="16x8x4"` が効くのが
  x GEMM 1 本であることとも整合する。
  **副次的だがこちらの方が重要な発見**: `nstep=20` は **cuBLAS を含む経路の比較には
  短すぎる**。cuBLAS の 1 回限りの初期化が **19.1 ms** あり、`nstep=20` の Main の
  **42%** を占める（`CUDA_MODULE_LOADING=EAGER` でも CUDA Graph 再生でも消えない）。
  `nstep=100`/`400` の 2 点回帰で定常の傾きを採り直すと、
  **`CUDAFORTRAN_GEMM`（709.2 µs/stage）は `GEMM_CUTE`（794.9）より速く**、
  `nstep=20` で見えていた順位（1003 対 811）は**逆だった**。
  これで nsys のカーネル和と実行時間が µs 単位で一致する。
  定常状態の同一 DOF 4 点は **269.3 / 346.4 / 590.0 / 990.7 µs/stage**、
  FP64 ピーク比 **13.0 → 15.0 → 28.8 → 65.7%**。p=63 は p=255 の **1.68 倍**速い。
  続けて残る 3 つを当たったが**全部効かない**（`p63_gap_study.md` §11）:
  `volume_flux` の mainloop 融合は、残存実装が `cp.async` を捨てる形なので
  p=255 の 1.66 倍という結果が次数に依存せず効く（加えて x を CUTLASS に戻す
  −68 µs を先に払う必要があり、1 方向あたりの算術も −37 対 +55 µs で合わない）。
  z GEMM のタイル / stage 段数の掃引は現行の `<64,32>`/4 が最良で、
  `flux_z` の読み出しが半分になる `<64,64>` ですら **+6.9%**。
  y GEMM のワープ形状 / 段数は最大 0.6% で再実行のばらつきと同じ。
  **測定衛生の注意**: ログインノードの GPU は共有で、他人のプロセスが載っていると
  同じ baseline が 592 → 761 µs になる。`nvidia-smi --query-compute-apps` で確認し
  `CUDA_VISIBLE_DEVICES` で空き GPU に固定すること。
  **`Escale` を x/y epilogue に移す案も不採用（§12）**: 実装前にアブレーションで
  上限を測ったところ、**移せる読み出しを全部消しても 2.6%**（−402 MB / −3 ストリームで
  591.7 → 576.3 µs/stage）。実際の案はバイトを消さず移すだけなのでそれ以下になる。
  この測定で 2 つの思い込みが壊れた: **「z のオペランド 1 本 47 µs」は p=63 では
  約 5 µs**（p=255 からの外挿が 10 倍外れていた）、そして
  **z は epilogue のオペランド帯域で律速されていない**（バイトの 43% を消して
  カーネル時間の 7%）。lift の再構成と 6 面 gather を消すと**効果はゼロ**、
  y epilogue の 2 要素/アクセス化は +1〜4%。融合 epilogue が素の z GEMM に足している
  +125 µs は、ロードでも lift でもなくアキュムレータのステージングと出力経路にある。
  **測定法**: `nstep=100`/`400` の 2 点回帰は n=100 側で ±5% 振れる。
  `nstep=400` 単独（cuBLAS の切片 19 ms は全変種共通）なら ±0.1%。
- **p=127, `Ne=2³`（2026-08-27）**: 同一 DOF の 5 点目で、**演算強度がマシンバランスを
  越える最小の次数**。最速は **`CUDAFORTRAN_GEMM_FUSED`**、Main **2.3941 ms/step** /
  **710.3 µs/stage**（graph on 2.3383、−2.3%）。
  **（訂正 2026-08-28、`p127_gap_study.md` §11）最速は `CUDAFORTRAN_FUSED_TC` に交代した。**
  **（訂正・2026-08-28）その後 `GEMM_FUSED` の z epilogue 改修で最速は
  `CUDAFORTRAN_GEMM_FUSED` に戻った**（752.8 対 `FUSED_TC` 787.8 µs/stage、
  `p127_gap_study.md` §12）。以下は当時の記述である。
  この日 p=127 に融合カーネルを開き、**707.7 µs/stage / Main 2.3969 ms/step** で
  `GEMM_FUSED`（711.3 / 2.4051）を上回った。5 回ずつの中央値でレンジは重ならない。
  以下の GEMM 経路についての記述と掃引結果は当時のまま有効である。`CUDAFORTRAN_GEMM` の 1.16 倍、
  `CUDAFORTRAN_SPLIT` の 8.8 倍。**コードは 1 行も要らなかった**: p=63 の作業で入れた
  LGL 演算子生成・`case default` フォールバック・CUTLASS batch 上限検査がそのまま通し、
  `mod_mesh.f90` の `dense_lift = (Nq <= 128)` も最初から Nq=128 を含んでいた。
  演算強度は **5.34 FLOP/B** で**マシンバランス 5.08（40.1 TFLOP/s ÷ 7.9 TB/s）の
  向こう側**にあり、`p63_gap_study.md` §9 の「次に越えるのは p=127」という予測が当たった。
  越えるのが p=127 が初めてという意味ではなく、**越える次数のうち最小のものが p=127**
  ということである（5 点の演算強度は 0.88 / 1.53 / 2.55 / **5.34** / 10.32 で、
  p=255 はとうに越えている側。交差は p=63 と p=127 の間、`Nq ≈ 122`）。
  **ただし越え幅は 5% でリッジ点の真上**であり、達成効率も演算 46.5% / 帯域 44% と拮抗して
  どちらの屋根にも当たっていない。変わったのは屋根ではなく改善の効く方向である。
  達成効率は **18.63 TFLOP/s（46.5%）/ 3.49 TB/s（44%）**で、同一 DOF 5 点の FP64 ピーク比は
  **13.0 → 15.0 → 28.7 → 46.5 → 65.8%** と単調に上がる（帯域比は 75 → 50 → 57 → 44 → 32% と下がる）。
  **（追記 2026-08-27）この単調性は 6 点目を入れると崩れる。**（**さらに訂正: 崩れていなかった。**
  p=31 に `FUSED_TC` を書いた後は 24.1% で、6 点は単調である —— `p31_gap_study.md` §14.2。
  以下は `FUSED_TC` を書く前の記述である。）p=31 は 12.5% で 6 点中最小で、
  p=15 と p=63 の両方より遅い。`p31_gap_study.md` §8 を参照。
  **「GEMM 経路の転送量は Nq に依存しない」が 3 点目でも成立**（147.8 B/ノード。
  `volume_flux_kernel` が p=63/127/255 で 128.9 / 129.0 / 129.3 µs、実行命令数まで
  18,350,080 で同一）。**次数依存ノブを 4 種掃引して採用ゼロ**:
  x GEMM を cuBLAS に渡す案は **p=127 で既に互角**（差 0.3%）で `Nq <= 64` の閾値は
  そのままが正しい（p=63 の 1.59 倍という差は `K=64` に固有だった）、
  `CutlassMmaShape` は `8x8x4` 維持、側ストリームは on が正しい（−1.0%）、
  `GemmZ` のタイル / 段数は 3 次数とも動かない。
  副産物として **`16x8x16` の損の一部は `TileK=32` そのもの**であることが分かった
  （`8x8x4` のまま TileK だけ 32 にすると +33%）。詳細は `p127_gap_study.md`。
  **追加掃引（同日）**: `GemmX` のスレッドブロックタイルはどの次数でも未掃引だったので
  p=127 で振った。**`M = Nq = 128` をタイル 1 枚で覆って `flux_x` の読み出しを半減させても
  効かない**（711.5 対 709.7 µs）—— x GEMM は DRAM 11% の SM 発行律速なので当然である。
  `GemmY` のタイルも同様に効かない。これで CUTLASS 側のノブ 5 種は全部掃引済みで採用ゼロ。
  **なぜ何も効かないのかの説明も付いた**: ncu の occupancy limiter を見ると
  x / y / z の 3 本とも **`Block Limit Registers` と `Block Limit Shared Mem` が完全に一致**
  している（2/2、3/3、4/4）。CUTLASS の段数がその釣り合う点に置かれているので、
  **片方だけ削っても常駐ブロックは 1 つも増えない**。
  p=127 に残る手は GEMM 経路の外側だけで、最大のものは
  **tendency の外の RK 更新カーネル（Main の 9.4%、1 step 1.5 GB）**だが、
  これを z epilogue に畳むのは `dqdt` を実体化しないということであり、抽出インタフェースを
  特殊化してしまう。実運用として今日できるのは `UseCudaGraph = .true.` の −2.3% のみ。
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

- **p=31 `FUSED_TC` の残り天井を測って探索を終了した（2026-08-30、
  `p31_gap_study.md` §18–19）**: 採用ゼロ。xz は **`lg_throttle`**（L1/TEX 79%、
  占有率 25%）、y は **`mio_throttle`**。面 2,4（i 固定）を消すと −17.8% だが、
  ループ内評価も先行平面ステージも **+25.6%**、体積端の遅延評価は ±0 または
  +31.5%。占有率 25%→50% はスピルで +25.0%、D1D 再読で +28.9%。面 gather の
  `ld.global.cs` は +17.5%。1 要素 1 ブロックは +7.9%、末尾バリア前送りは
  +0.58%、`Escale` の前出しは +2.3%、y の `sDQ` スウィズルは ±0。最速は
  `CUDAFORTRAN_FUSED_TC` のまま（占有 GPU 357.6 µs/stage）。
- **p=31 Tensor Core 融合（2026-08-27、`p31_gap_study.md` §14）**: **p=31 の最速は
  `CUDAFORTRAN_FUSED_TC`**、Main **1.38328 ms/step** / **374.8 µs/stage**
  （graph on 1.34960、−2.4%）、FP64 ピーク比 **24.1%**。CUDA core 融合版の **2.66 倍**、
  `CUDAFORTRAN_GEMM` の 1.67 倍（tendency）。**（追記 2026-08-30）CC 側を §20 で
  718.2 µs まで詰めたので、いまの主比は 359.7 / 718.2 = 2.00 倍。最速は TC のまま。**
  shared wavefront が **7.6〜9.1 分の 1**
  になったのが機構で、手書き TC がこれまで出した 1.21（p=7）/ 1.17（p=15）倍を
  大きく超える。Nq=32 では **x と z の出力添字が同じ (i,k)** なので z の shared 往復が
  存在せず、転置形では D1D フラグメントが j に依存しないのでレジスタに載る。
  **これにより下の 2 つの結論が否定された**: 「p=31 は曲線の極大点」（同一バイナリで
  最速経路どうしを並べると 273.0 → 349.8 → **374.8** → 597.1 → 707.5 µs/stage で
  **単調**）と「融合が勝つ次数の上限は p=15」（CUDA core の融合に限った話だった）。
- **p=31, `Ne=8³`（2026-08-27、上記より前）**: 同一 DOF の 6 点目にして最後の点。**当時の最速は素の
  `CUDAFORTRAN_GEMM`**、Main **2.4403 ms/step** / **725.3 µs/stage**（graph on 2.3582、−3.3%）。
  `GEMM_FUSED` の 1.18 倍、`CUDAFORTRAN_SPLIT` の 3.4 倍速い。**`GEMM_FUSED` ではなく素の
  `GEMM` が勝つのは同一 DOF 族でここだけ**である。
  **p=31 は曲線の極大点である**（**訂正: §14.2 で否定された**）: 同一ビルド・同一 `nstep` で 434.8（p=15、CUDA core
  `FUSED`）→ **725.3（p=31）** → 593.8（p=63）µs/stage で、**両隣より遅い**。
  同一 DOF で p=63 の半分の演算量しか無いのに p=63 より 1.22 倍遅く、FP64 ピーク比
  **12.5% は 6 点中最小**。5 点で見えていた単調性はここで崩れる。
  理由は「**Nq 非依存の転送量を半分の演算量で払う**」ことで、ncu では
  `dqdt_assembly` / `volume_flux` / `elembnd_flux` の 3 本が DRAM 81〜91% の屋根に
  張り付き、GEMM 自体も `K=32` の浅さで占有率 24〜30% しか出ない。
  **「GEMM 経路の転送量は Nq に依存しない」が 4 点目でも成立**（`volume_flux_kernel` が
  p=31 で 125.9 µs、同一実行ファイルの p=63 で 129.3 µs）。
  **Nq=32 の融合カーネル 2 本を新規に書いた**（`tendency_fused_p31_xz_kernel` /
  `_y_kernel`、`CUDAFORTRAN_FUSED` を p=31 に開いた）。x と z が同じスレッド写像
  `(i,k)` を共有し `j` を内側ループにすると累算器がスカラで済むという構造で、
  shared 42.5 / 16 KB、レジスタ 64 / 44、スピル 0、バンクコンフリクト 0.1〜0.6%。
  **結果は 1001.7 µs/stage で GEMM に 1.38 倍負けた。** 負けた理由は実装ではなく
  次数の性質で、**融合の転送量優位が p=7 の 3.7 倍から 1.13 倍（133.4 対 148 B/ノード）
  まで縮み**、カーネルは L1/TEX 84%（DRAM 18%、SM 39%）で詰まる —— 縮約を shared 経由で
  回すため **shared wavefront 1 本あたり 22 FLOP** にしかならない。
  したがって **融合が勝つ次数の上限は p=15 のままで、交差は p=15 と p=31 の間**にある
  （p=63 と p=31 の間ではない）。**（訂正: これは CUDA core の融合に限った話だった。
  Tensor Core の融合は p=31 でも勝ち、交差は p=31 と p=63 の間である —— §14.1。）**
  **副産物**: 面フラックスを j ループの外に括り出すと **−22.2%**（1287.4 → 1001.7 µs）。
  総面点数は不変で、変わったのは並列度だけ（ループ内では 1024 スレッド中 32 本しか
  生きていない分岐を 32 回まわしていた）。**CUDA Fortran の `__shfl` は 1-based** で、
  0-based のつもりで書いた版は相対差 2.08e-16 → 7.29e-15 の劣化として検証に出た。
  **訂正**: `p63_gap_study.md` §9 と `p127_gap_study.md` §10 の「p=31 は
  `Nq*Ne = 262144` なので CUTLASS 経路が使えない」は誤りで、正しくは
  `32*512 = 16384`。`GEMM_FUSED` / `GEMM_CUTE` とも無改造で走る（実測）。
  次数依存ノブは 6 種掃引して**採用ゼロ**で、cuBLAS x-GEMM 閾値 `Nq <= 64` と
  側ストリーム閾値 `Nq >= 64` がそれぞれ 4 点で裏付けられた。詳細は `p31_gap_study.md`。

- **`GEMM` / `GEMM_CUTE` 経路の lift+assembly 融合（2026-08-27、`p31_gap_study.md` §13）**:
  `separable_lift_kernel` が書く `lift_out(Np,Ne)` は次の `dqdt_assembly_kernel` 以外
  誰も読まないので、実体化するだけで **1 stage あたり 268 MB**（書き 134 + 読み 134）を
  捨てていた。分かれていたのは `CUDAFORTRAN_SPLIT` が両者を別々に計時するためだけである。
  `separable_lift_assembly_kernel` に統合し、2 本で 251.5 µs だったものが **160.9 µs**。
  **`CUDAFORTRAN_GEMM` は p=31 −11.8%、p=63 −12.0%、p=127 −10.3%、p=255 −7.5%**、
  `GEMM_CUTE` は p=31 で −8.8%。**絶対削減量は 4 次数で 93.8〜96.2 µs/stage とほぼ一定**で、
  消した 268 MB が同一 DOF なら次数に依らないことと一致する（相対値だけが縮むのは
  `K = Nq` が深くなって分母が伸びるため）。
  総和順序を保ったので**全次数・全経路でビット一致**。`CUDAFORTRAN_SPLIT` は従来のまま。
  なお最初の測定は p=255 を「変化なし」と記録したが、それは
  `conf_perf_p255.conf` が `GEMM_FUSED` を選ぶため素の `GEMM` を測っていなかった
  ためで、誤りである（`p31_gap_study.md` §13.2 の注記）。
  **`GEMM_FUSED` は不変** —— CUTLASS の z epilogue が同じ融合を既に持っており、
  それが `GEMM_FUSED` が p≥63 で速かった理由の一部だった。最速経路はどの次数でも
  変わらないが、p=63 / p=127 で `GEMM_FUSED` 対 `GEMM` の差は **1.20 倍から 1.04 倍**に縮んだ。
  **`FUSED_TC` を Nq=32 に開く件は未実装で、当初の見送り理由は誤りだった**
  （`p31_gap_study.md` §13.5）。「GB200 では FP64 TC ピークが CUDA core ピークと
  等しいので書く根拠が無い」と書いたが、この融合カーネルは演算律速ではない。
  測り直すと **L1/TEX を詰まらせているのは shared が 87〜91%**（global ではない）で、
  1 shared wavefront あたり 21.6 FLOP。m8n8k4 の mma は 128 FLOP/wavefront なので
  **律速そのものを 5.9 倍緩める**見込みがあり、機構としては TC 書き換えを支持する。
  一方で `GEMM` の 631 µs/stage を上回るには **1.59 倍**が要るのに対し、
  手書き TC の前例は p=7 で 1.21、p=15 で 1.17 倍にとどまる（mma は命令数を削るが
  レジスタを減らさず、占有率 50% の壁が残るため）。**律速の所在は支持、前例は否定**で、
  賭けとしては分が悪いと判断して本作業では書いていない。
  **不採用**: `elembnd_flux` を `volume_flux` に融合して M 側 gather を消す案は、
  アドレスパターンだけを変えたアブレーションで**上限 −0.7%** と分かり着手しなかった
  （片側だけ coalesced にしても効かず、両側で初めて −8.4%。P 側は近傍要素を指すので
  coalesced にできない）。

- **融合カーネルを p=127 に開き、p=127 と p=63 の最速を更新した（2026-08-28、
  `p127_gap_study.md` §11、`p63_gap_study.md` §18）**:
  `CUDAFORTRAN_FUSED` / `CUDAFORTRAN_FUSED_TC` が残っていなかった唯一の次数を埋め、
  **`FUSED_TC` 707.7 µs/stage で `GEMM_FUSED`（711.3）を抜いて p=127 の最速になった**。
  初稿の 1141.2 から 5 段階（1.60 → 1.22 → 1.10 → 1.065 → 1.042 → **0.995 倍**）で、
  数値の意味は全過程で不変（検証 4.16e-16、spill 0、最後の 2 段はビット一致）。
  **設計**: 融合 2 本とも 128×64 タイル（縮約添字 m は全域）、1024 スレッド、
  1 ワープ 2×2 mma タイル。m を全域に保つと `sD` 1 枚が両オペランドに効く。
  プリフェッチは **`cp.async` が使えない**（パネルが持つのは `q*u` という積で
  コピーではない）ので、**1 平面 1 回しか読まないフラックスパネルだけを全深さで
  常駐**させ、2 倍冗長な方と L2 常駐の `D1D` をチャンクする。
  **ncu（Slurm 62173 / 62193）で律速を確定させた**:
  **Tensor Core 化がしたことは律速を shared から DMMA パイプへ移すこと**である
  （CUDA core 版は mio throttle 13.1 ワープ・shared ld 201.3 M ウェーブフロント・
  l1tex 86.3%、TC 版は 0.16・33.6 M・58.5%）。「mma を消しても変わらないから
  演算律速ではない」という当初の推論は**誤り**で、置換先の `DFMA` は同じ
  "Shared"（FP64/Tensor）パイプを使う。
  そこから 2 つの手が出た。**(1) スウィズルの代数的簡約**: mma ループ内では
  畳む場が `colk`（レーン定数）なので `swt(行 + stride·l) = (行 ^ colk·4) + stride·l`
  とループ不変になり、SASS の整数命令が 501 → 168、全命令 122.5 M → 86.6 M（−29%）、
  **ビット一致で −2.0%**。**(2) バリアの配置**: チャンクループ本体の末尾に置いた
  `__syncthreads` は最終反復の後に走るだけなので、先頭へ `if (kk)` 付きで移すと
  1 本消える。**ビット一致で p=127 −4.5%（4 本→3 本）、p=63 −7.7%（2 本→1 本、
  `BK63 = NQ63` でループが 1 回しか回らないため丸ごと無駄だった）**。
  p=255 は逆効果（+10.7%）で棄却 —— **効果はチャンク数で決まる**。
  **モデル（§11.12）**: どの資源も飽和していない（DMMA パイプ 53.1%、l1tex 58.5%、
  DRAM 12.6%、occupancy 49.4%、issue 24.3%）。5 種類の削減
  （FLOP −89% → 0%、shared ロード −25% → 0%、命令数 −29% → −2.0%、
  フラックス global ロード 全部 → −7%、**バリア −25% → −4.5%**）のうち
  **スケールしたのはバリアだけ**で、律速は資源ではなく **32 ワープが同期点で
  揃うのを待つ時間**である。占有率 50% はレジスタ（1024×64 = レジスタファイル全部）と
  shared（192 KB）が同時に 1 ブロックを指していて動かせない。
  **交差点は p=127 と p=255 の間へ動いた**: 同一実行ファイルで
  p=15 / 31 / 63 / 127 はすべて `FUSED_TC`（325.8 / 378.9 / 493.4 / 707.1）、
  p=255 のみ `GEMM_FUSED`（1487.4 対 955）。**融合が勝つ上限は少なくとも p=127** で、
  「上限は p=63 で確定」と書いた §11.4 の初稿は §11.11 より前の測定に基づく誤りだった。

## 読むときの注意

各レポートは書かれた時点の commit と Slurm job を明記している。後の変更で
結論が変わった箇所には追記を入れてあるが、**表の数値そのものは当時の値のまま**
なので、commit 表記を確認すること。

最適化が守るべき数値契約はリポジトリ直下の `AGENTS.md` にある。
