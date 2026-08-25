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
