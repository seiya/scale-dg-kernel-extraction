# Ozaki Scheme I 本体実装レポート（`CUDAFORTRAN_GEMM_OZAKI1`）

作成日: 2026-08-28
対象リポジトリ: `scale-dg-kernel-extraction`（worktree `scale-dg-kernel-extraction-ozaki`）
ブランチ / HEAD: `feature/ozaki` / `e237fbb` 上の **未コミット作業ツリー**（本レポートと Ozaki I 本体）
関連実装: [`ozaki2_implementation_report.md`](ozaki2_implementation_report.md)（Scheme II）
対象 GPU: NVIDIA GB200（RIKYU login ノード）、`make CUDA=1 GPUFLAGS=-gpu=cc100`
ビルド: NVIDIA HPC SDK 26.3、cuBLAS 13.2.1

本レポートは **volume 導関数 GEMM を Ozaki Scheme I で置換する経路**を本体に統合した記録である。
Scheme II（`CUDAFORTRAN_GEMM_OZAKI2`）と並ぶ **比較・計測用経路**であり、性能最適化の主目的ではない。
`CublasEmulation`（cuBLAS 内蔵 fixed-point）とは独立した経路である。

---

## 1. 実装概要

| 項目 | 内容 |
|---|---|
| 新カーネル種別 | `CUDAFORTRAN_GEMM_OZAKI1`（`DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1 = 10`） |
| CUDA コア | `cuda_ozaki1_gemm.cu` — A・B 両方のスライス分解、最大 s² 本の INT8 `cublasGemmEx`、CRT **なし** |
| 統合 | `cuda_cal_dqdt_gemm_ozaki1`（`mod_cuda_dg_kernels.cuf`）— 既存 `cuda_cal_dqdt_gemm` と同段構成 |
| namelist | `OzakiSliceCount`（2–16、既定 8）。`OzakiModuliCount` は OZAKI2 専用 |
| 検証 conf | `input_p7_val_gemm_ozaki1.conf`、`input_p255_val_gemm_ozaki1.conf` |

### 1.1 Scheme I と Scheme II の違い

| 観点 | Scheme I（本実装） | Scheme II（OZAKI2） |
|---|---|---|
| 分解 | **A と B の両方**を INT8 スライス（最大 s 枚ずつ） | **A のみ**マルチスライス、B は 1 枚 |
| 整数 GEMM 本数 | 最大 **s_a × s_b**（方向あたり） | s 本 + **Garner CRT** で 1 方向分を再構成 |
| 再構成 | `scale_a[i] × INT32 × scale_b[j]` を **FP64 で直接加算** | 法ごと residue → CRT → FP64 |
| namelist ノブ | `OzakiSliceCount` | `OzakiModuliCount` |
| 典型ワークスペース | `iB` が s × Np×Ne（B 側スライス） | `residues` が s × Np×Ne×4 B |

パイプライン（volume 部分）は OZAKI2 と同じ:

```
volume_flux_kernel
  → ozaki1_dgemm / ozaki1_dgemm_strided_batched（×3 方向）
  → separable_lift_assembly_kernel
```

`AGENTS.md` の数値契約: `flux_*`・`D1D`・`D1D_tr` は点ごと配列のまま。`Escale` の重み付けは
再構成後の assembly で適用する（OZAKI2 と同順序）。

---

## 2. アルゴリズム（`cuda_ozaki1_gemm.cu`）

1. **B の分解**（`decompose_b_nn`）: 列 max / 127 で量子化 → 残差が閾値超なら次スライス（最大 `OzakiSliceCount`）。
2. **A の分解**（`decompose_a_tn`）: 行 max、pack は TN 形（OZAKI2 と同じ cuBLAS 呼び出し `transa=T`）。
3. **スライスペアループ**（`run_slice_pairs`）: 各 `(i,j)` で INT8 GEMM → `ozaki1_recon_*` が
   `C += scale_a[i,row] × prod × scale_b[j,col]`（最初のペアのみ上書き、以降加算）。
4. **同期**: 各 `ozaki1_dgemm*` 末尾で `dg_cuda_stream` を synchronize。

INT32 飽和: 内積次元 K = Nq ≤ 256 なので 1 回の INT8 GEMM あたり |Σ| ≤ K×127² ≈ 4.1×10⁶
（INT32 上限 2.1×10⁹ に対し余裕。p=255 でも K=256 のため安全）。

---

## 3. 実装時に修正した欠陥

### 3.1 `scale_a` バッファ不足（z 方向 GEMM）

**症状**: p=7 で `dqdt` が NaN / 10⁴ オーダーの外れ値、`OzakiSliceCount=2` では
`ozaki1_dgemm z-deriv` で `CUDA_ERROR_ILLEGAL_ADDRESS`。

**原因**: `scale_a` を `slices × Nq × max_batch` で確保していたが、z 導関数は
`m = nq²` 行（p=7 では 64 行）。batched row-max が `scale[b×m + row]` に書き、
Nq=8 分の領域を **8 倍オーバーラン**。

**修正**: `max_m = Nq²` を workspace に持ち、`scale_a` とスライス間オフセットを
`slices × max_m × max_batch` に拡張。OZAKI2 も同型のインデックスを使うが、
OZAKI2 は slice あたり scale を上書き再利用するため顕在化しにくかった。

### 3.2 ゼロ行のスケール

**症状**: 上記修正前は定数速度 p=7 で q が ±1.8×10³⁰⁸ に発散。

**原因**: max=0 の行に `scale=1.0` を置き、`scales_need_next_slice` が
「127 > 1.0」で真となり、**無意味な追加スライス**（最大 s² ペア）を量産。

**修正**: ゼロ行は `scale=0`、pack 時は `denom>0` のときのみ除算。

---

## 4. OZAKI2 との比較（設計・トレードオフ）

| 観点 | OZAKI1 | OZAKI2 |
|---|---|---|
| cuBLAS INT8 呼び出し | O(s_a × s_b) / 方向 | O(s) + CRT カーネル |
| B（大きな flux）の分解 | **毎方向・毎ステージ** | 1 枚 pack のみ |
| 精度メカニズム | 浮動小数加算でスライス和 | 複数法 CRT（GEMMul8 整合 moduli） |
| 参照実装との対応 | 論文 Scheme I の素直な写像 | [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) Scheme II に近い |
| p=7 単体 GEMM（合成データ） | x/y/z max diff ≈ **3.3 / 1.9 / 1.4×10⁻³** | **同一値**（同一 INT8 核） |
| p=7 全体 `dqdt` vs native | max abs **21.5** | **23.7** |

単体 GEMM テスト（`bench_ozaki2/mini_ozaki2_all.cu` を Ozaki I API に差し替え）では
OZAKI1/2 が **同一誤差** —— 差は **B 側マルチスライス本数**と **CRT vs FP64 加算**に出る。

---

## 5. 数値検証

login ノード、`nstep=1`、定数速度（`SCALE_DG_VARYING_COEFF` 未設定）。
`SCALE_DG_DUMP_DQDT` で owned `dqdt(:,1:Ne)` 全点比較、参照は `CUDAFORTRAN_GEMM`。

| ケース | max abs diff vs native | 備考 |
|---|---:|---|
| p=7, Ne=32³, s=8 | **2.15×10¹** | OZAKI2（2.37×10¹）と同オーダー。傾向は一致、量子化誤差 |
| p=7, s=2 | **2.15×10¹** | スライス数を減らしても max は同程度（飽和は主に 1–2 スライスで足りるケースが多い） |
| p=255, Ne=1, s=8 smoke | q ≈ ±1.0 | 発散なし。`dqdt` 全点比較は未実施（ファイルサイズ 16M 点） |

`SCALE_DG_VARYING_COEFF=1` では OZAKI1/2 とも native 比 max abs が 10¹ オーダーになり、
**Ozaki 経路全体の精度特性**（volume GEMM 量子化）を示す。定数速度ベンチだけでは
見えない外れ値は、変動係数テストで捕捉する（`AGENTS.md`）。

---

## 6. 性能（参考、login ノード・nstep=1）

device イベント行（`CUDA device Ozaki-I GEMM` / `GEMM tendency`）を 1 step 分として記録。
**login ノード共有 GPU のためばらつきあり**。傾向比較用。

| 経路 | p=7 Ne=32³ | p=255 Ne=1 |
|---|---:|---:|
| `CUDAFORTRAN_GEMM` | 39.6 ms | 24.0 ms |
| `CUDAFORTRAN_GEMM_OZAKI1`（s=8） | **127 ms（≈3.2×）** | 15.1 ms |
| `CUDAFORTRAN_GEMM_OZAKI2`（s=14） | 73 ms（≈1.8×） | 48.4 ms |

p=7 では **s² まで増える INT8 GEMM** のため OZAKI2 より遅いのが典型。p=255 Ne=1 の 1 回測定では
OZAKI1 が native より速く見えたが、スライスが早期打ち切りされた場合の **ノイズ**と見なし、
本番最速経路としては採用しない。

---

## 7. メモリ（workspace 概算）

setup ログ例:

```
# p=7 Ne=32768
ozaki1 workspace: Nq=8 Ne=32768 Np=512 slices=8 iB=134.2 MB

# p=255 Ne=1
ozaki1 workspace: Nq=256 Ne=1 Np=16777216 slices=8 iB=134.2 MB
```

| バッファ | サイズ（s = OzakiSliceCount） |
|---|---|
| `iB` | s × Np × Ne（INT8） |
| `iA` | s × Nq² × max(Nq×Ne, Ne)（INT8） |
| `scale_a` | s × **Nq²** × max_batch × 8 B（§3.1 修正後） |
| `scale_b` | s × Np × Ne × 8 B（非 batched B 最大列数は nq²×Ne だが batched 小 B は先頭のみ使用） |
| `prod` | Np × Ne × 4 B |
| `res_a`, `res_b` | Np × Ne × 8 B |

p=255 Ne=1、s=8 では `scale_a` だけ **約 1.0 GiB**（Nq²×256×8×8）。OZAKI2 の
residue 940 MB と同オーダーだが、**s² 回 GEMM** の方が時間側のペナルティが大きい。

---

## 8. 統合上の制約

- **CUDA Graph 非対応**: ホスト側 slice 判定・可変 s_a/s_b・stream sync のため
  `advect3d_eq_graph_supported` が false（OZAKI2 と同様）。
- **p=255**: `PolyOrder=255` ゲートに OZAKI1 を追加済み。
- **Ne 汎用**: x/y/z の strided batched は OZAKI2 と同じ stride 契約（`strideB=0` で
  共有 B）。`Ne>1` smoke は p=7 で実施。

---

## 9. 再現方法

```bash
module load nvhpc
export CUTLASS_HOME=/path/to/third_party/cutlass
cd scale-dg-kernel-extraction-ozaki
make clean && make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100

# smoke
./scale-dg_extraction input_p7_val_gemm_ozaki1.conf
./scale-dg_extraction input_p255_val_gemm_ozaki1.conf

# dqdt 比較（p=7）
./scale-dg_extraction input_p7_val_gemm.conf
export SCALE_DG_DUMP_DQDT=dqdt_ref.txt
./scale-dg_extraction input_p7_val_gemm.conf
export SCALE_DG_DUMP_DQDT=dqdt_ozaki1.txt
./scale-dg_extraction input_p7_val_gemm_ozaki1.conf
# paste + awk で max abs diff（約 21）
```

非 CUDA ビルド: `make clean && make` — stub に OZAKI1 シンボルあり。

---

## 10. まとめ

- **Ozaki Scheme I** を `CUDAFORTRAN_GEMM_OZAKI1` として volume GEMM 3 本に統合。
  CRT なし・A/B 両スライス・FP64 直接加算が Scheme II との本質差。
- **クリティカル修正**: z 方向の `m=Nq²` に合わせた `scale_a` 確保と、ゼロ行スケール処理。
  これ無しでは p=7 で数値破壊・GPU 非法アクセス。
- **性能結論**: p=7 では native の **約 3 倍**、OZAKI2 の **約 1.8 倍遅い**（s=8）。
  最速経路ではないが、Scheme I の **計測可能な正しい経路**として残す。
- **今後**: 単体 `ozaki1_crt_test` 相当の bench、変動係数での p=255 全点比較、
  B 側スライス早期打ち切りのプロファイル（実効 s_a/s_b のログ）は任意。

関連: [`ozaki2_implementation_report.md`](ozaki2_implementation_report.md)、
[`ozaki2_survey_2504.08009.md`](ozaki2_survey_2504.08009.md)、
[`cublas_emulation_survey.md`](cublas_emulation_survey.md)、
[`AGENTS.md`](../AGENTS.md)。
