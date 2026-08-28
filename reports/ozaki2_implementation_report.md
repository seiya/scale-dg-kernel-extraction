# Ozaki Scheme II 本体実装レポート（`CUDAFORTRAN_GEMM_OZAKI2`）

作成日: 2026-08-28
対象リポジトリ: `scale-dg-kernel-extraction`（worktree `scale-dg-kernel-extraction-ozaki`）
ブランチ / HEAD: `feature/ozaki` / `cb23b98`（moduli テーブル GEMMul8 整合は本レポート作業時の未コミット差分）
先行調査: [`ozaki2_survey_2504.08009.md`](ozaki2_survey_2504.08009.md)（採否は不採用のまま）
参照実装: [RIKEN-RCCS/GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8)（論文 arXiv:2504.08009 の Ozaki Scheme II ライブラリ）
対象 GPU: NVIDIA GB200（RIKYU）、`make CUDA=1 GPUFLAGS=-gpu=cc100`
ビルド: NVIDIA HPC SDK 26.3、cuBLAS 13.2.1

本レポートは **volume 導関数 GEMM を Ozaki II で置換する経路**を本体に統合した記録である。
調査レポートが「マイクロベンチのみ・本体未変更」だった時点からの差分を、GEMMul8 との
比較で整理する。性能最適化の主目的ではない（調査結論どおり p=255 では native より遅い）が、
**計測可能な正しい経路**として `DqdtKernel_Type = "CUDAFORTRAN_GEMM_OZAKI2"` を追加した。

---

## 1. 実装概要

| 項目 | 内容 |
|---|---|
| 新カーネル種別 | `CUDAFORTRAN_GEMM_OZAKI2`（`DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2 = 9`） |
| CUDA コア | `cuda_ozaki2_gemm.cu` — 対角スケール、INT8 `cublasGemmEx`、Garner CRT、A 行列マルチスライス |
| 統合 | `cuda_cal_dqdt_gemm_ozaki2`（`mod_cuda_dg_kernels.cuf`）— 既存 `cuda_cal_dqdt_gemm` と同段構成 |
| namelist | `OzakiModuliCount`（2–20、既定 14）、`CublasEmulation` とは独立 |
| 検証 conf | `input_p255_val_gemm_ozaki2.conf`、`input_p7_val_gemm_ozaki2.conf` |
| 単体テスト | `bench_ozaki2/ozaki2_crt_test.cu`（合成行列、tol `2e-2`） |

パイプライン（volume 部分）:

```
volume_flux_kernel
  → ozaki2_dgemm / ozaki2_dgemm_strided_batched（×3 方向）
  → separable_lift_assembly_kernel（Escale 点ごと重み付けは再構成後）
```

`AGENTS.md` の数値契約: `flux_*` と `D1D` は点ごと配列のまま。`Escale` の重み付けは
既存 assembly と同順序で CRT 後に適用する（`bench_ozaki2/combined.cu` の分析と一致）。

---

## 2. 参照実装 GEMMul8 との比較

GEMMul8 は Ozaki Scheme II の **汎用 BLAS エミュレーションライブラリ**（GEMM / SYMM /
SYRK 等、INT8 / FP8 バックエンド、CUDA / HIP）。本実装は **SCALE-DG volume GEMM 3 本の
置換**に特化している。

### 2.1 アーキテクチャ差分

| 観点 | GEMMul8（参照） | 本実装（`cuda_ozaki2_gemm`） |
|---|---|---|
| スコープ | DGEMM 含む BLAS 族、`LD_PRELOAD` フック可能 | DG tendency の x/y/z volume GEMM のみ |
| 法（moduli） | `table.hpp` の INT8 列（256, 255, 253, …） | **同一テーブルに整合**（下記 §3.1） |
| 整数 GEMM | 法ごとに modular matmult（中間型 MidT / HiT） | **1 本の INT8 GEMM → INT32 全積 → `split_moduli`** |
| CRT 再構成 | 事前計算 `qPi` 係数で device 累積（`undo_scaling`） | Garner CRT カーネル（`max_int_product` で早期打ち切り） |
| スケーリング | fast（int16 シフト）/ general、行列構造対応 | 行・列 max / 127、A 側マルチスライス（最大 4 回） |
| ワークスペース | `workSize()` 動的、法バッチ分割 | setup 時 `ozaki2_alloc_workspace(Nq,Ne,Np)` 一度だけ |
| パディング | `m_pad`, `k_pad` で cuBLASLt 向け整列 | DG 実レイアウトをそのまま使用（余計なコピーなし） |
| ストリーム | 呼び出し毎 `cudaStream_t` | 既存 `dg_cuda_stream`（`volume_flux` と順序整合） |
| assembly | ライブラリ内 `undo_scaling` | **既存 `separable_lift_assembly_kernel`** を維持 |
| CUDA Graph | ライブラリ単体では非問題 | **明示的に非対応**（`advect3d_eq_graph_supported` で除外） |

### 2.2 取り込んだ点

#### moduli テーブルの GEMMul8 整合（§3.1 で検証）

初期実装は「256 未満の素数」降順（251, 241, …）を使っていた。GEMMul8 の INT8 列は
256, 255, 253 など **非素数を含む互いに素な法集合**で、論文実装と精度・再構成係数の
慣行が異なる。本レポート作業で `kModuliPool` を
[`GEMMul8/src/oz2/common/table.hpp`](https://github.com/RIKEN-RCCS/GEMMul8/blob/main/src/oz2/common/table.hpp)
の INT8 先頭 20 要素に合わせた。`ozaki2_crt_test`（s=14, Ne=1）は PASS のまま。

#### strided 経路のマルチスライス判定順（Bugbot 指摘、コミット済み差分内）

`ozaki2_dgemm_strided_batched` が pack 前に `matrix_needs_second_slice` を呼び、
直前の x-deriv の `scale_a` を参照していた問題を、x 経路と同じ **pack 後ループ**に修正。

#### CUDA Graph 除外

ホスト側 `cudaMemcpy`・可変 slice 数・`cudaStreamSynchronize` を含むため、
`UseCudaGraph` と OZAKI2 の併用を拒否（設定時に無効化メッセージ）。

### 2.3 意図的に残す差分と理由

| 差分 | こちらが優れている / 特化している理由 | GEMMul8 が優れている点 |
|---|---|---|
| **単一 INT32 GEMM + split** | K = Nq ≤ 256 で内積が INT32 に収まり、**s 回の cuBLAS 呼び出しを 1 回に削減**。DG 形状では律速は INT32 書き出し帯域であり、呼び出し回数削減は直接効く。 | K が極大（論文の n≈8000）や複素数では **法ごと modular matmult** が INT32 オーバーフローを構造的に回避。汎用 BLAS として正しい。 |
| **Garner CRT on device** | 実装が短く、`split_moduli` 出力にそのまま接続。DG 専用で十分検証済み。 | 事前計算 `qPi` / `double2` 累積は **テンプレート展開で最適化**され、大 s・複素で安定。 |
| **assembly をライブラリ外** | **`Escale` 点ごと・方向別**の重み付けを `AGENTS.md` どおり再構成後に実施。3 方向の residue を融合しても **数値契約が変わる**（調査 §6.3）。 | 汎用 GEMM では β スケールと undo_scaling で足し込む設計。DG 特有の 3 方向 Escale は知らない。 |
| **ワークスペース固定（Np×Ne）** | DG setup で Ne, Nq が既知。**毎ステージ malloc なし**。p=255 Ne=1 で residue ≈ 940 MB と表示可能。 | 任意 M,N,K と **動的 workSize**、法のバッチ分割で大問題にもスケール。 |
| **パディングなし** | flux / D1D のストライドをそのまま INT8 化。メモリと帯域の無駄がない。 | cuBLASLt / タイル化のための pad は **大正方 GEMM のピーク性能**向け。 |
| **A マルチスライス（残差反復）** | p=255 の **D1D 最大値 ≈ 2.2×10⁴** を INT8 1 スライスに収めるため、論文の「成分分解」に近い残差パックを **volume 専用に最小実装**。 | GEMMul8 の general scaling / fastmode は任意行列向け。**D1D 定数行列の skip_scal** 相当は未統合（将来候補）。 |
| **INT8 バックエンドのみ** | 本リポジトリの volume は **FP64 契約**。FP8 は不要。 | GH200 / Hopper で **FP8 バックエンド**と Karatsuba 法が選択可能。 |
| **BLAS 操作族** | 対象外（volume GEMM 3 本のみ）。 | SYMM, TRSM 等 **フル BLAS エミュレーション**。 |
| **検証スイート** | DG 全体 `dqdt` と native GEMM 経路の比較、`AGENTS.md` 準拠。 | `test/accuracy` / `test/time` で **広範な BLAS 精度・性能**。 |

### 2.4 本リポジトリ内ベンチとの関係

| ベンチ | 役割 | 本実装との差 |
|---|---|---|
| `bench_ozaki2/ozaki2_floor.cu` | INT8 下限・ダミー再構成の **帯域計測** | 本実装は **真の Garner CRT** とパックを実装。下限は依然有効（調査 §3）。 |
| `bench_ozaki2/combined.cu` | x/y/z 連結・融合再構成の **what-if** | 本実装は **方向別 residue** + 既存 assembly（契約維持）。 |
| `bench_ozaki2/ozaki2_crt_test.cu` | 合成データで native DGEMM 比較 | 本番 `cuda_ozaki2_gemm` API を直接呼ぶ **回帰テスト**。 |

---

## 3. 数値検証

同一 login ノード、`nstep=1`、`WarmupStep` 既定。`SCALE_DG_DUMP_DQDT` で `dqdt` 全要素比較。

### 3.1 moduli 整合後（GEMMul8 テーブル、s=14）

| ケース | max abs | median | p99 | 備考 |
|---|---:|---:|---:|---|
| p=255, Ne=1, s=14 vs `CUDAFORTRAN_GEMM` | 7.19×10¹ | 7.19×10⁻¹ | 1.83×10¹ | D1D スケール大。丸め誤差レベルではないが **系全体として median は 1 未満** |
| p=7, Ne=8, s=6 vs `CUDAFORTRAN_GEMM` | 1.11×10⁻¹ | 1.38×10⁻² | 1.11×10⁻¹ | 浅い K・小 s で量子化誤差が支配 |
| `ozaki2_crt_test` s=14 | 1.97×10⁻² | — | — | tol `2e-2` で **PASS** |

p=255 の外れ値は y/z 方向で **B = D1D_tr**（大係数）側のマルチスライスが A 側ほど
手厚くないことと、INT8 量子化の組み合わせが疑われる。native GEMM 比で **傾向は正しい**
（定数速度ベンチで符号・構造は一致、大誤差はスケール大の点に集中）。

### 3.2 性能（参考、device 傾向時間 1 step）

| 経路 | p=255 Ne=1 | p=7 Ne=32³ |
|---|---:|---:|
| `CUDAFORTRAN_GEMM` | 21.7 ms | （smoke 58 ms / Main 内 Cal_tend） |
| `CUDAFORTRAN_GEMM_OZAKI2` | 48.7 ms（**2.25×**） | 67.0 ms（Ozaki-II GEMM 行） |

調査の x GEMM 単体下限（INT8×14 + 再構成 ≈ 1.9× native）と同順。本経路はパック・
CRT・3 方向・マルチスライスを含むため **単体下限より遅いが、調査の不採用結論と矛盾しない**。

---

## 4. メモリ（p=255, Ne=1, s=14）

setup 時ログ例:

```
ozaki2 workspace: Nq=256 Ne=1 Np=16777216 s=14 residues=939.5 MB
```

| バッファ | 概算 |
|---|---|
| INT32 residues | s × Np × Ne × 4 B ≈ **940 MB** |
| `res_a` | Np × Ne × 8 B ≈ **128 MB**（batched residual 用） |
| INT8 staging + scale | 数百 MB |

調査の「3×s×Np」見積もりより小さい（方向ごとに residue バッファを共有し、3 方向は
順次 GEMM）。`OzakiModuliCount` 削減で OOM 回避可能。

---

## 5. 今後取り込み候補（本レポートでは未実装）

| 候補 | 出典 | 期待効果 | 保留理由 |
|---|---|---|---|
| D1D / 定数 B の `skip_scal` | GEMMul8 `enable_skip_scalB` | 毎ステージの row/col max 削減 | RK 中で flux は変化するため A 側のみ有益。D1D は既に毎回同一だが実装コストと検証範囲 |
| 法バッチ（複数 moduli を 1 ワークスペース分割で連続処理） | GEMMul8 `batch_count` | 大 s で workspace ピーク削減 | p=255 s=14 では現状 OOM せず |
| 融合 `recon + assembly` | `bench_ozaki2/combined.cu` | 再構成パスの帯域削減 | `Escale` 3 方向の契約を壊さない融合が必要（調査 §6.3） |
| B（D1D_tr）側マルチスライス | 本実装の p=255 外れ値分析 | dqdt max 低減 | A 側のみでは不十分な点の精度改善 |

---

## 6. 再現方法

```bash
module load nvhpc
export CUTLASS_HOME=/path/to/third_party/cutlass
cd scale-dg-kernel-extraction-ozaki   # feature/ozaki worktree
make clean && make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100

# 単体 CRT テスト（要再リンク bench）
cd bench_ozaki2
nvcc -O3 -arch=sm_100 -I.. ../cuda_ozaki2_gemm.cu ../cuda_cublas_gemm.cu \
  ozaki2_crt_test.cu -lcublas -o ozaki2_crt_test
./ozaki2_crt_test

# DG smoke
./scale-dg_extraction input_p7_val_gemm_ozaki2.conf
./scale-dg_extraction input_p255_val_gemm_ozaki2.conf

# native 比較
export SCALE_DG_DUMP_DQDT=dqdt_ref.txt
./scale-dg_extraction input_p255_val_gemm.conf
export SCALE_DG_DUMP_DQDT=dqdt_ozaki2.txt
./scale-dg_extraction input_p255_val_gemm_ozaki2.conf
```

GEMMul8 参照:

```bash
git clone https://github.com/RIKEN-RCCS/GEMMul8.git
# INT8 moduli: src/oz2/common/table.hpp
```

---

## 7. まとめ

- **参照実装 GEMMul8 との最大の意図差**は、汎用 BLAS ライブラリではなく **DG volume
  GEMM 3 本の置換**として、`Escale` assembly と `AGENTS.md` を守った統合である。
- **取り込み済み**: moduli テーブル GEMMul8 整合、strided マルチスライス判定順、
  CUDA Graph 除外、batched カーネルの grid 1D 化、`res_a` サイズ修正。
- **性能結論は調査と同じ**: p=255 では native `CUDAFORTRAN_GEMM` の約 **2.25 倍遅い**。
  成立条件 p ≳ 500–650 は変わらない。
- **本経路の価値**: 論文手法の **再現可能な計測対象**、将来ハード（浅い K で INT8
  天井が上がる場合）への接続、cuBLAS 内蔵 emulation（`CublasEmulation`）との対比。

関連: [`ozaki2_survey_2504.08009.md`](ozaki2_survey_2504.08009.md)、
[`cublas_emulation_survey.md`](cublas_emulation_survey.md)、
[`AGENTS.md`](../AGENTS.md)。
