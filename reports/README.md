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

## 現時点の結論

- **p=7, `Ne=32^3`**: `CUDAFORTRAN_FUSED_TC` が最速（commit `e22dda1` 以降）。
  それ以前は `CUDAFORTRAN_FUSED` が最速だった。
  さらに occupancy を 100% に上げる作業（`tc_paper_survey_2407.09621.md` §7）で
  device 時間 1.076 → 0.851 秒。同じ知見を CUDA core 版にも適用すると
  そちらも 1.153 → 0.986 秒になり（§8）、両者を 100% occupancy で揃えた
  TC 版の優位は 1.16× である。
- **p=255, `Ne=1`**: `CUDAFORTRAN_GEMM_FUSED` が最速。手書きの Tensor Core 経路は
  CUTLASS / cuBLAS の multistage mainloop に大きく負ける。
- 同じ体積 DOF 数でも、p=7 と p=255 で最適戦略は逆転する。
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
- **TMA は FP64 では既製品が無い（2026-08-25、調査のみ）**: CuTe は `double` の
  tensor map を作れる（`copy_sm90_desc.hpp:220`）が、CUTLASS 4.7 の collective
  builder に FP64 特殊化は 1 つも無く、SM90 の builder が前提にする wgmma に
  FP64 が無い。TMA + DMMA の mainloop は手書きなら書けるが、x/y GEMM は既に
  FP64 ピークの 86–88% なので天井は上がらない。狙えるとすれば z GEMM の
  レジスタ圧だけ。詳細は `overall_summary_report.md` §12.9。
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
