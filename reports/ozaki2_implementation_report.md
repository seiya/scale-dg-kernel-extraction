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

**注記（2026-09-09、Slurm `88634`）: ここでいう「正しい」は経路役割の意味に限る
—— `AGENTS.md` の未融合 GEMM ドライバ上で volume GEMM 3 本だけを差し替える、という
経路定義を満たすという意味であって、数値精度の合格を意味しない。`OZAKI2` は
`gemm_assignment_and_carrier.md` §14.3 で 3 次数・4 設定（FIXED / ADP × 7 / 14 moduli）
すべて✘の数値不合格である**（FIXED での max abs は p=7 で 2.97e-01、p=255 で
1.235e+02、p=511 で 2.118e+02。対照は p=7 が `CUDAFORTRAN_SPLIT`（§9.8.5）、
p=255 / 511 が同一次数の `CUDAFORTRAN_GEMM`（§14.2）。moduli を 7 から 14 に
増やしても値は同一で、律速は moduli 数ではなく A 側の int8 表現）。**本経路の測定値は「数値が合わない設定の
速度」として読むこと。**

---

## 1. 実装概要

| 項目 | 内容 |
|---|---|
| 新カーネル種別 | `CUDAFORTRAN_GEMM_OZAKI2`（`DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2 = 9`） |
| CUDA コア | `cuda_ozaki2_gemm.cu` — 対角スケール、INT8 `cublasGemmEx`、Garner CRT、A 行列マルチスライス |
| 統合 | `cuda_cal_dqdt_gemm_ozaki2`（`mod_cuda_dg_kernels.cuf`）— 既存 `cuda_cal_dqdt_gemm` と同段構成 |
| namelist | `OzakiModuliCount`（2–20、**既定 7**。本レポート作業時の既定は 14 で、`fd091fc` で 7 に変更された）、`CublasEmulation` とは独立 |
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
./scale-dg_extraction namelists/val_p7_gemm_ozaki2_ne32.conf
./scale-dg_extraction namelists/val_p255_gemm_ozaki2.conf

# native 比較
export SCALE_DG_DUMP_DQDT=dqdt_ref.txt
./scale-dg_extraction namelists/val_p255_gemm.conf
export SCALE_DG_DUMP_DQDT=dqdt_ozaki2.txt
./scale-dg_extraction namelists/val_p255_gemm_ozaki2.conf
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

**注記（2026-09-09）: 上の「成立条件 p ≳ 500–650」は交叉していない。** 本レポート
§8.13 の 3 台測り直し（commit `302c681`）では GB200 の対 `GEMM` 比が p=255 で 2.9×、
p=511 で 1.7×、**p=767 でも 1.3× で 1 を割らない**（H100 は 2.3〜3.1×、A100 は
1.9〜3.1× で下げ止まり —— 最小は H100 が p=511 の 2.3×、A100 が p=511 / 575 の
1.9×）。
[`README.md`](README.md) の公表列（job `78846`、現行既定 FIXED 55 / 7 moduli）でも
GB200 p=255 は **3.03×**（3 190.9 / 1 052.0）で、p=767 の **1.30×**
（82 149.1 / 63 286.1）が全 9 次数の下限である。すなわち
**測れた範囲（p≤767）では、どの機械でも native `GEMM` を下回る次数は現れていない**。

**注記（2026-09-09、Slurm `88634`）: 上の「計測可能な経路」としての価値は、
経路役割の意味に限る。** `OZAKI2` は `gemm_assignment_and_carrier.md` §14.3 で
p=7 / 255 / 511 の 3 次数、FIXED / ADP × 7 / 14 moduli の 4 設定すべてが✘の
**数値不合格**である（FIXED で p=7 2.97e-01、p=255 1.235e+02、p=511 2.118e+02。
対照は p=7 が `CUDAFORTRAN_SPLIT`、p=255 / 511 が同一次数の `CUDAFORTRAN_GEMM`）。
本節の性能値は数値が合わない設定の速度であり、合格経路との性能比較には使えない。

関連: [`ozaki2_survey_2504.08009.md`](ozaki2_survey_2504.08009.md)、
[`cublas_emulation_survey.md`](cublas_emulation_survey.md)、
[`AGENTS.md`](../AGENTS.md)。


---

## 8. sm_90 / sm_80 で 1 次数も走らない —— 診断と候補修正（2026-09-07）

`measurement_inventory.md` §21.2 が、**H100（sm_90）と A100（sm_80）では
`CUDAFORTRAN_GEMM_OZAKI2` が全次数で落ちる**ことを測った
（SGE `8592913` / TCS `9538052`、commit `a464048`）。**sm_100 では同じツリーで
9 次数 48 ラン全て通る**（Slurm `83482`、§18.7）。症状は 2 台・全次数で同一:

```
Ozaki-II: FIXED 7 moduli (CRT product 55.7 bits, single A pack)
ozaki2 workspace: Nq=128 Ne=8 Np=2097152 s=7 residues=469.8 MB
 ozaki2_dgemm x-deriv: an illegal memory access was encountered
Accelerator Fatal Error: ... error 700 (CUDA_ERROR_ILLEGAL_ADDRESS)
```

### 8.1 事実 —— `iB` は論理サイズちょうどで確保されている

- x 微分の呼び出しは
  `ozaki2_dgemm(0, 0, m=Nq, n=Nq²·Ne, k=Nq, D1D, lda=Nq, flux_x, ldb=Nq, ...)`
  （`mod_cuda_dg_kernels.cuf`）。**B は `k·n = Nq·Nq²·Ne = Np·Ne` バイト**である。
- 確保は `g_ws.iB_bytes = mn`、`mn = Np·Ne`（`cuda_ozaki2_gemm.cu`
  `ozaki2_alloc_workspace`）。**バイト単位で完全一致し、スラックがゼロ。**
- 対して `iA` は `k_rows·max_batch = Nq²·max(Nq·Ne, Ne)` で、x 微分が要る
  `k·m = Nq²` を大きく上回る。**きつく確保されているのは `iB` だけである。**
- ワークスペース確保自体は成功している（上のログの 469.8 MB が出ている）。
  **メモリ不足ではない。**
- `cuda_ozaki2_gemm.cu` には `__CUDA_ARCH__` もアーキ分岐も共有メモリも
  cluster も無い（`grep` で 0 件）。**アーキで変わるのは、`cublasGemmEx` が
  選ぶ INT8 カーネルだけである。**

### 8.2 仮説（**測定ではない**）

cuBLAS の INT8（IMMA）カーネルはタイルをベクトル化して読み、**論理的な
末尾を越えて読んでからマスクする**実装が一般的である。sm_100 で選ばれる
カーネルはその越境をしないか、越境幅が確保の外に出ない。sm_90 / sm_80 で
選ばれるカーネルは出る —— `iB` の直後には何も無いので、そこで
`CUDA_ERROR_ILLEGAL_ADDRESS` になる。

**これは仮説である。** GB200 では再現しないので本レポートは確かめられない。
確かめるのは §8.4 の再走である。

### 8.3 候補修正 —— INT8 バッファに 512 バイトのスラック

```c
const size_t kInt8Slack = 512;
g_ws.iA_bytes = static_cast<size_t>(k_rows * max_batch) + kInt8Slack;
g_ws.iB_bytes = static_cast<size_t>(mn) + kInt8Slack;
```

**行列の定義は 1 つも変えていない** —— `m` / `n` / `k` / `lda` / `ldb` / `ldc`
はそのままで、確保だけが伸びる。したがって**演算は不変**であるべきで、
実際 GB200 で確かめた:

| 検証 | 結果 |
|---|---|
| p=63 `OZAKI2`、`SCALE_DG_VARYING_COEFF=1`、owned `dqdt` 全 16 777 216 点、修正前バイナリと比較 | **ビット一致** |

**GB200 での挙動は変わらない**（修正前も修正後も走る）。
**スラック 512 は根拠のある値ではなく、越境幅の上限を知らないので
安全側に取った定数である。**

### 8.4 これを確かめる測定（**まだ走っていない**）

**予測（この節を書いた時点で凍結）**:

- **O1（決定的）**: 修正後の H100 と A100 で、`OZAKI2` が **全次数で
  `rc=0`** になる。**ならなければ §8.2 の仮説は誤りで、
  越境読み以外の原因を探すことになる。**
- **O2**: GB200 の `OZAKI2` の時間は修正前と**測定誤差内**で一致する
  （確保が 512 バイト増えるだけなので）。
- **O3**: 修正後の H100 / A100 の `OZAKI2` の `dqdt` は、同じ機械の
  `CUDAFORTRAN_GEMM` に対して §18.7 が GB200 で記録したのと同程度の
  差に収まる。

手順は `jobs/tsubame_ozaki.sh` / `jobs/wisteria_ozaki.sh` を、
この修正を含むツリーで**ビルドし直してから**投げ直すだけである。


### 8.5 予測 O1 は外れた —— **パディングでは直らない**（2026-09-07、SGE `8594961` / TCS `9539267`）

§8.3 の修正を含む commit **`3b66356`** で両機を走らせ直した。

| 機械 | ジョブ | `rc=0` | 失敗 | 修正前（`a464048`） |
|---|---|---:|---:|---|
| H100 | SGE `8594961` | 93 | **51** | 93 / 51 |
| A100 | TCS `9539267` | 87 | **48** | 87 / 48 |

**成功数も失敗数も 1 つも変わらない。**症状も同一である:

```
ozaki2 workspace: Nq=8 Ne=32768 Np=512 s=7 residues=469.8 MB
 ozaki2_dgemm x-deriv: an illegal memory access was encountered
```

**したがって §8.2 の仮説（cuBLAS の INT8 カーネルが `iB` の末尾を越えて読む）
は否定された。** §8.3 のパディングは**残す** —— GB200 でビット一致であり
害が無く、`iB` が論理サイズちょうどであったこと自体は事実だからだが、
**これは原因ではなかった。**

### 8.6 次に測ること —— **推測をやめて `compute-sanitizer` を当てる**

**`ozaki2_dgemm x-deriv` は「エラーが気づかれた場所」であって
「起きた場所」ではない。** CUDA のエラーは非同期で、`ozaki2_dgemm` は
`launch_col_max` → `launch_pack_b` → `launch_row_max` → `launch_pack_a` →
`cublasGemmEx` の順に**自前カーネル 4 本を先に投げてから** GEMM を呼ぶ。
**違反はそのいずれで起きていてもおかしくない**（実際、致命エラー自体は
さらに後の `update_halo` で出ている）。

`compute-sanitizer --tool memcheck` は**カーネル名とアドレスを名指しする**。
`jobs/tsubame_ozaki_sanitizer.sh` / `jobs/wisteria_ozaki_sanitizer.sh` を
用意した。**p=7 を 2 つの規模で試す** —— memcheck は 10〜100 倍遅いので
`val_p7_gemm.conf`（`Ne=2³`）を先に、再現しなければ
`perf_p7_gemm.conf`（`Ne=32³`）を走らせる。**小さい構成を使うのは
クラッシュの診断だからであって、性能比較では入力を変えない**という
`AGENTS.md` の規則はここには掛からない。

**本節は原因をまだ特定していない。**


### 8.7 原因 —— **CRT のテーブルをホストポインタのままカーネルに渡していた**（`compute-sanitizer`、SGE `8595140`、node `r18n4`）

`compute-sanitizer --tool memcheck` が 1 回で名指しした。**cuBLAS ではない。**

```
========= Invalid __global__ read of size 4 bytes
=========     at ozaki2_split_moduli_kernel(int *, const int *, const int *, int, long long)+0x270
=========     by thread (32,0,0) in block (12,0,0)
=========     Access to 0x11fb9a4 is out of bounds
=========     and is 22,475,117,184,604 bytes before the nearest allocation
=========         Host Frame: launch_split_moduli(long long)
=========         Host Frame: ozaki2_dgemm
=========         Host Frame: ... mod_cuda_dg_kernels.cuf:2496
========= ERROR SUMMARY: 4100 errors
```

`Ne=2³` でも `Ne=32³` でも同じカーネル・同じアドレス（`0x11fb9a4`）で、
**規模に依らない**。

**アドレス `0x11fb9a4` は実行ファイルのデータ領域である。**
`Ozaki2State g_state` は**ホスト側の static 構造体**で、
`int moduli[kMaxModuli]` と `int64_t garner_inv[kMaxModuli]` は
**その中のホスト配列**である。それを 3 か所でカーネル引数にそのまま渡していた:

| 起動 | 渡していたホスト配列 |
|---|---|
| `launch_split_moduli` | `g_state.moduli` |
| `launch_crt` | `g_state.moduli`, `g_state.garner_inv` |
| `launch_crt_batched` | `g_state.moduli`, `g_state.garner_inv` |

**なぜ GB200 でだけ動いていたか。** GB200 は Grace CPU と Blackwell GPU が
NVLink-C2C でキャッシュコヒーレントなので、**カーネルからホストのアドレスを
参照できてしまう**。TSUBAME（x86 ＋ H100）と Wisteria（x86 ＋ A100）では
できないので faults する。**「sm_90 / sm_80 で動かない」ではなく、
「GB200 でだけ、たまたま動いていた」が正しい。**

**§8.2 の仮説（cuBLAS の INT8 カーネルの越境読み）が外れた理由も明確である**
—— 落ちていたのは GEMM ではなく、その**前に投げられる自前カーネル**だった。
`ozaki2_dgemm` が返すエラーは非同期に検出されたもので、
**発生場所を示していなかった**（§8.6）。

### 8.8 修正 —— テーブルをデバイスに置く

`Ozaki2State` に `int *d_moduli` と `int64_t *d_garner_inv` を足し、
`ozaki2_init` で `cudaMalloc` ＋ `cudaMemcpy`（`precompute_garner_inverses`
の直後、値が確定してから）、`ozaki2_finalize` で解放。
**3 か所の起動をデバイス側ポインタに差し替えた。**

**値は同じで、置き場所だけが変わる。** GB200 で確認した:

| 検証 | 結果 |
|---|---|
| p=63 `OZAKI2`、点変化係数、owned `dqdt` 全 16 777 216 点、修正前バイナリと比較 | **ビット一致** |

**予測（凍結）**:

- **O1′（決定的）**: 修正後の H100 と A100 で `OZAKI2` が**全次数 `rc=0`** に
  なる。**ならなければ、この 3 か所以外にも同種の箇所が残っている**か、
  別の原因がある。
- **O2′**: GB200 の時間は修正前と測定誤差内で一致する（テーブルの読み出しが
  ホストからデバイスに変わるだけで、`s` = 7 要素の参照である）。
- **O3′**: `compute-sanitizer` を修正後に当てると
  `ozaki2_split_moduli_kernel` のエラーが消える。**残るエラーがあれば
  それが次の対象**である。

**同種の欠陥が他のファイルに無いかは確認していない**（`cuda_ozaki1_gemm.cu`
など）。**GB200 だけで開発すると、この種の誤りは検出されない**という
一般的な含意がある。


### 8.9 同種の欠陥の全数監査 —— **他には無い**（2026-09-07、静的、追加測定なし）

§8.7 の欠陥は「**カーネルがデリファレンスするポインタがホストメモリを指す**」
という型で、**GB200 のコヒーレンシがそれを隠す**。他に無いかを全ファイルで
調べた。**追加測定は無い。**

**対象: `.cu` 7 ファイルのカーネル起動 62 か所と、`.cuf` の 27 か所。**

| ファイル | 起動 | 結果 |
|---|---:|---|
| **`cuda_ozaki2_gemm.cu`** | 13 | **3 か所で該当**（`launch_split_moduli` / `launch_crt` / `launch_crt_batched`）。§8.8 で修正済み |
| `cuda_ozaki1_gemm.cu` | 11 | **該当なし** —— 11 起動とも引数は呼び出し元から来たデバイスポインタか `g_ws.res_a`（`cudaMalloc`）。`Ozaki1State` / `BDecompCache` / `Ozaki1SliceStats` は**配列メンバを持たず**、カーネルに渡してもいない |
| `cuda_dg_kernels_tc.cu` | 18 | **該当なし** |
| `cuda_dg_kernels_fused_highp.cu` | 10 | **該当なし** |
| `cuda_cutlass_gemm_fused.cu` | 9 | **該当なし** |
| `cuda_dg_kernels_fused.cu` | 1 | **該当なし** |
| `cuda_cublas_gemm.cu` | 0 | 起動なし（cuBLAS 呼び出しのみ） |

- **ファイルスコープのホスト配列をカーネルに渡している箇所は、修正した 3 つ
  以外に無い。** 検索でヒットした「ファイルスコープ配列」は**すべて
  `__global__` 関数内のローカル配列**（`double acc[2*TM*TN]` など）だった。
- **`&ホスト変数` をカーネル引数にしている起動も無い。**
- **`cudaLaunchKernelEx` の `cudaLaunchConfig_t` / `cudaLaunchAttribute` は
  該当しない** —— これらはドライバがホスト側で読む起動設定であって、
  カーネルがデリファレンスするものではない。
- **CUDA Fortran 側（27 起動）はコンパイラが検出する。**
  `attributes(global)` サブルーチンの実引数には `device` 属性が要求されるので、
  ホスト配列を渡せばビルドが通らない。**チェックの効かない経路は C++ 側だけ
  だった。**

**GB200 のコヒーレンシが隠しうる隣接する型も見た**:

| 型 | 結果 |
|---|---|
| `cudaMemcpy` の向き | **10 か所すべて明示**（D2H 8、H2D 2）。`cudaMemcpyDefault` は 0 |
| managed / host-registered / mapped メモリ | **使用なし**（`cudaMallocManaged` / `cudaHostRegister` / `cudaHostAlloc` / `cudaMallocHost` とも 0） |

**したがって、この型の欠陥は `cuda_ozaki2_gemm.cu` の 3 か所だけであり、
修正済みである。**

**ただしこれは静的監査であって、`compute-sanitizer` の再走ではない。**
**予測 O3′（修正後にサニタイザからこのエラーが消える）が経験的な確認であり、
そこで別のエラーが出れば、この監査が見ていない型があったということである。**


### 8.10 コヒーレント読みの代金は GB200 では測れない —— **§18.7 の公表値は影響を受けない**（2026-09-07、Slurm `85290`、node `c087`、`--exclusive`）

**「GB200 はホストメモリを C2C 越しに取りに行っていたのだから、性能劣化に
なっていたのではないか」**という問いを測った。**なっていなかった。**

修正前（ホスト側テーブル、`e113535` 期の凍結コピー `scale-dg_extraction.s33`）と
修正後（デバイス側テーブル、`98b5f34`）を、同じ namelist・`OZAKI2` のみ・
占有ノードで交互に走らせた。物差しは
`CUDA device Ozaki-II GEMM` ÷ (計時 step 数 × 3)。

| p | n | 修正前（ホスト）[µs/stage] | 修正後（デバイス） | 差 | レンジ |
|---:|---:|---:|---:|---:|---|
| 7 | 6 | 5 201.3 | 5 196.1 | −0.10% | 重複 |
| 63 | 6 | 3 310.9 | 3 304.8 | **−0.18%** | **非重複** |
| 255 | 6 | 3 177.4 | 3 175.1 | −0.07% | 重複 |
| 511 | 3 | 23 224.8 | 23 212.2 | −0.05% | 重複 |
| 767 | 3 | 81 698.2 | 81 820.0 | **+0.15%** | **非重複** |

**登録した予測は 2 つとも外れた**（`output/s35/PREDICTIONS.txt`）:

- **P1（デバイス版はどの次数でも遅くならない）は外れ** —— p=767 で
  **+0.15%、レンジ非重複**で**遅い**。
- **P2（少なくとも 1 次数で 1% 超の改善）は外れ** —— 最大の差は
  p=63 の **0.18%** で、1% に遠く届かない。
- P3 は「改善があれば」という条件付きなので発火しない。

**機構: テーブルが小さすぎる。** `moduli` は `int` 7 個、`garner_inv` は
`int64_t` 7 個で、**合わせて 84 バイト**である。全スレッドが読むが、
**最初の 1 回で L2 に載ればあとは行き先がホストでもデバイスでも同じ**で、
C2C の往復は `mn` 回ではなく実質 1 回しか起きない。

**したがって `measurement_inventory.md` §18.7 が公表した GB200 の Ozaki 列
（job `83482`、commit `e113535`、ホストポインタ版）は、この修正の影響を
受けない。** 書き換える必要は無い。

**p=767 の +0.15% は測定値として記録する。**機構は追っていない
（`cudaMalloc` が 2 回増えたことでアドレス配置が動いた可能性はあるが、
**確かめていない**）。

**この結果を一般化しないこと。** 示されたのは「**84 バイトの読み取り専用
テーブル**をコヒーレントに取っても GB200 では測れない」であって、
コヒーレントアクセス一般が無料だということではない。**修正の理由は
性能ではなく正しさである** —— H100 と A100 では、この同じアクセスが
プログラムを落としていた（§8.7）。


### 8.11 予測 O3′ は両機で成立した —— **メモリエラーは消えた**（SGE `8595166`、TCS `9539920`）

修正（`98b5f34`）を含む commit で `compute-sanitizer --tool memcheck` を
当て直した。

| 機械 | ジョブ | node | 構成 | 修正前 | **修正後** |
|---|---|---|---|---:|---:|
| H100 | SGE `8595166` | `r17n1` | `Ne=2³` | 4 100 errors | **0 errors** |
| H100 | 〃 | 〃 | `Ne=32³` | 45 572 errors | **0 errors** |
| A100 | TCS `9539920` | `wa35` | `Ne=2³` | （未取得） | **メモリエラー 0** |
| A100 | 〃 | 〃 | `Ne=32³` | （未取得） | **メモリエラー 0** |

**両機・両構成とも `Invalid __global__ / __shared__ / __local__` は 1 件も
出ない。**`ozaki2_split_moduli_kernel` のエラーは消えた。
**exit code も 0**（修正前は 1）。

**A100 に残る `ERROR SUMMARY: 1 error` は memcheck の検出ではない**:

```
Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument"
  on CUDA API call to cuCtxGetLimit.
    Host Frame: cublasLtCtxInit -> cublasCreate_v2 -> cublas_gemm_init
    Host Frame: mod_advect3d_eq.f90:270 -> main.f90:314
```

**cuBLAS が自身の初期化で制限値を問い合わせて失敗を握りつぶしている
内部プローブ**であり、`setup_advect3d_eq_setup` の時点、すなわち
**Ozaki の処理が始まる前**に 1 回だけ出る。本レポートのコードでも、
§8.8 の修正とも無関係である。H100 側（nvhpc 26.1）では出ず、
A100 側（nvidia/25.9、`libcublas.so.12`）でだけ出るので、
**cuBLAS のバージョン差**と読める。**追っていない。**

**§8.9 の静的監査（「この型の欠陥は他に無い」）は、両機で経験的に
裏づけられた。** サニタイザが見る型の中では、他に何も残っていない。

### 8.12 まだ判定できていないこと —— **O1′**

**予測 O1′（修正後に `OZAKI2` が全次数 `rc=0` になる）はまだ検定されていない。**
手元に届いている `jobs/*_ozaki.sh` の結果は commit **`3b66356`**
（パディングのみの版）のもので、H100 93/51・A100 87/48 のままである。
**§8.8 の修正を含むビルドでの再走が要る。**

**サニタイザが 0 件になったことは O1′ を強く示唆するが、O1′ そのものでは
ない** —— サニタイザは p=7 の 2 構成しか見ておらず、`jobs/*_ozaki.sh` は
9 次数 × 3 経路 × 6 ラウンドを走らせる。


### 8.13 予測 O1′ は成立した —— **`OZAKI2` が 3 台で走るようになった**（SGE `8595165`、TCS `9539837`）

修正（`98b5f34`）を含む commit **`302c681`** で `jobs/*_ozaki.sh` を再走した。

| 機械 | ジョブ | `rc=0` | 失敗 | 修正前（`3b66356`） |
|---|---|---:|---:|---|
| H100 | SGE `8595165` | **141** | **3** | 93 / 51 |
| A100 | TCS `9539837` | **132** | **3** | 87 / 48 |

**残る失敗は 3 本ずつで、いずれも既知の OOM である** ——
H100 は p=767 の `OZAKI1`（94 GB）、A100 は p=575 の `OZAKI1`（40 GB）で、
**`OZAKI2` の失敗はゼロ**。§21.3 が記録したメモリ量どおりの挙動であり、
欠陥ではなく結果である。

**`OZAKI2` は 3 台すべてで全次数走るようになった**（A100 は p=575 まで、
これはメモリ量の制約）。**`measurement_inventory.md` §19.7 の Ozaki 行は
これで閉じる。**

`OZAKI2` の対 `GEMM` 比と絶対値（`Step loop per stage`、µs、中央値）:

| p | GB200 | 比 | H100 | 比 | A100 | 比 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 5 198 | 3.1× | 8 366 | 2.8× | 12 369 | 2.2× |
| 15 | 5 074 | 5.6× | 5 288 | 2.5× | 11 665 | 3.1× |
| 31 | 3 557 | 5.1× | 4 796 | 2.7× | 8 087 | 2.8× |
| 63 | 3 340 | 4.9× | 4 789 | 2.9× | 8 376 | 3.0× |
| 127 | 3 399 | 4.3× | 5 049 | 3.1× | 8 591 | 3.0× |
| 255 | 3 212 | 2.9× | 4 713 | 2.8× | 7 953 | 2.3× |
| 511 | 23 361 | 1.7× | 37 922 | 2.3× | 69 852 | 1.9× |
| 575 | 34 100 | 1.6× | 60 725 | 2.5× | 112 135 | 1.9× |
| 767 | 81 109 | 1.3× | 169 659 | 2.5× | — | — |

- **`OZAKI2` は `OZAKI1` よりはるかに native に近い** —— 比は 3 台とも
  1.3〜5.6× で、`OZAKI1` の 9.4〜128.5×（§21.3）とは桁が違う。
- **GB200 だけ比が次数とともに単調に下がり、p=767 で 1.3× まで来る。**
  H100 は 2.3〜3.1× で下げ止まり、A100 も 1.9〜3.1× で止まる。
  **`OZAKI1` で見えたのと同じ形**（§21.3）である —— **emulation 系が
  native に近づくのは GB200 でだけ。**
- **どの機械でも、どの次数でも native `GEMM` より速くはならない。**
  `cublas_emulation_survey.md` §9 の FIXED 55 が p=511 / 767 で 1 を切るのとは
  対照的である（**別実装なので比較ではなく並置**）。

**§8.10 との整合**: この列は修正後のビルドで測ったが、§8.10 が
テーブル配置の代金を **−0.18%〜+0.15%** と測っているので、
**§18.7 の GB200 公表値（修正前）とも比較可能**である。
