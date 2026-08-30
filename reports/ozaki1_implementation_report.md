# Ozaki Scheme I 本体実装レポート（`CUDAFORTRAN_GEMM_OZAKI1`）

作成日: 2026-08-28
対象リポジトリ: `scale-dg-kernel-extraction`（worktree `scale-dg-kernel-extraction-ozaki`）
ブランチ / HEAD: `feature/ozaki` / `74f09bc`（§6.2・§7.2 の slice 統計はその上の未コミット差分）
関連実装: [`ozaki2_implementation_report.md`](ozaki2_implementation_report.md)（Scheme II）
参照実装:
- [RIKEN-RCCS/GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) — Ozaki I は **cuBLAS 内蔵 FP64 fixed-point**（`test/common/ozaki1.hpp`）
- [enp1s0/ozIMMU](https://github.com/enp1s0/ozIMMU) — Ozaki Scheme I の **オープン INT8 Tensor Core** 実装
- 本リポジトリ [`cublas_emulation_survey.md`](cublas_emulation_survey.md) — `CublasEmulation` 経路（cuBLAS EAGER、別比較対象）
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
| 検証 conf | `input_p7_val_gemm_ozaki1.conf`、`input_p255_val_gemm_ozaki1.conf`、長時間 `input_p7_val_gemm_ozaki1_1000.conf` |
| 単体テスト | `bench_ozaki2/ozaki1_crt_test.cu`（合成行列、tol `2e-2`） |

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

## 2. 参照実装との比較

Ozaki Scheme I の「本家」参照は **2 系統**ある。GEMMul8 リポジトリ本体は Scheme II 中心だが、
ベンチの `OS1-*` ラベルは **cuBLAS 13 の FP64 fixed-point emulation**（`CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` +
`cublasSetFixedPointEmulationMaxMantissaBitCount`）を指す。オープンソースでアルゴリズム全体が読めるのは
**ozIMMU**（INT8 Tensor Core 上の slice GEMM + FP64 再構成）である。本実装は ozIMMU に近い
**明示的 slice 分解**だが、スケーリング方式と再構成経路は DG 用途に合わせて簡略化している。

### 2.1 アーキテクチャ差分

| 観点 | GEMMul8 / cuBLAS Ozaki I | ozIMMU | 本実装（`cuda_ozaki1_gemm`） |
|---|---|---|---|
| 提供形態 | cuBLAS **クローズド** emulation API | スタンドアロン C++/CUDA ライブラリ | DG volume GEMM **3 本のみ** |
| スライス分割 | 内部（mantissa bit 列、ADP オプション） | 行ごと **指数 max** + mantissa ビット切り出し | 行/列 **値 max / 127** + **残差反復** |
| スライス本数 | `num_slice`（`mantissaBitCount = num_slice×8−1`） | `fp64_int8_3` … `_18` 固定、**auto 選択**可 | `OzakiSliceCount`（2–16、既定 8）+ **早期打ち切り** |
| INT8 1 スライス当たり bit | 8（cuBLAS 設定） | **`get_bits_per_int8(K)`**（K 依存、最大 7） | 量子化幅 **127**（実質 7 bit + 符号） |
| 整数 GEMM ペア | 内部（非公開） | **s×(s+1)/2**（mantissa 位置 `(i,j)` の三角和） | **s_a × s_b**（残差スライスの直積和） |
| 中間型 | 非公開 | INT32 GEMM → **指数シフト**で FP64 累積 | INT32 GEMM → **scale_a×scale_b** で FP64 直接加算 |
| CRT | なし（Scheme I） | なし | なし |
| 定数 B の再利用 | 内部最適化（非公開） | なし（呼び出し毎 split） | **B 分解キャッシュ**（同一 `(ptr,k,n,ldb)`） |
| ワークスペース | `ozaki1::workSize`（1024/128 pad + 8 GiB 定数） | handle 内動的再割当 | setup 時 `ozaki1_alloc_workspace(Nq,Ne,Np)` 固定 |
| 比較用既存経路 | `CublasEmulation`（本 repo、130× 遅） | なし | **`CUDAFORTRAN_GEMM_OZAKI1`**（3× 程度、計測可能） |

### 2.2 取り込んだ点

#### ゼロ行スケールと pack 除算（§4.2、本 repo 独自修正）

ozIMMU / cuBLAS いずれも「ゼロ行を scale=1 で次スライスへ回す」経路は持たない。
本実装で `scale=0` + `denom>0` ガードを入れ、**無意味な s² ペア量産**と NaN を除去した。

#### `scale_a` バッファ（z 方向 m=Nq²）（§4.1）

参照実装は任意 M を想定し、DG 特化の batched row 数は持たない。z 導関数の
`m=nq²` に合わせ `max_m=Nq²` を workspace に持つ修正は **本用途固有**。

#### 定数 `D1D_tr` の B 分解キャッシュ（本レポート作業）

y/z 導関数は `strideB=0` で同一 `D1D_tr` を共有する。参照（ozIMMU）は呼び出し毎に
`split_B` するが、GEMMul8 / cuBLAS が内部で行う **定数行列 skip** に相当する最適化を、
`(B,k,n,ldb)` キーで **INT8 pack + scale_b を再利用**する形で取り込んだ。
1 tendency あたり B 分解は x=`D1D` 1 回 + `D1D_tr` 1 回に削減（従来 3 回）。

#### 単体回帰テスト `ozaki1_crt_test.cu`

ozIMMU の `test/main_test.cu` / OZAKI2 の `ozaki2_crt_test.cu` に倣い、p=255 形状の
x/y/z GEMM を native DGEMM と比較する bench を追加（tol `2e-2`）。

#### スライス数と cuBLAS パラメータの対応（ドキュメント）

GEMMul8 `ozaki1.hpp`: `mantissaBitCount = num_slice × 8 − 1`、
`numSlices = ⌈(maxMantissaBitCount+1)/8⌉`。FP64 尾数 53 bit なら **7 スライスで十分**、
既定 `OzakiSliceCount=8` は **余裕 1 枚**（cuBLAS OS1-8 と同じ意味）。コード変更はせず
namelist コメントとして整合を取った。

### 2.3 意図的に残す差分と理由

| 差分 | こちらが優れている / 特化している理由 | 参照が優れている点 |
|---|---|---|
| **残差 max/127 分解**（ozIMMU の指数 bit 切り出しでない） | **点ごとに変化する flux**（`AGENTS.md`）にそのまま適用できる。実装が短く、OZAKI2 と pack 核を共有しやすい。 | ozIMMU / cuBLAS は **mantissa ビット位置**が明確で、理論上 s×(s+1)/2 ペアに削減可能。**K が巨大**な一般 GEMM で INT32 オーバーフロー回避と整合。 |
| **s_a × s_b 全ペア GEMM** | 残差スライス同士は **独立残差**のため直積和が正しい。K≤256 では INT32 飽和余裕（§3）。 | ozIMMU の **s(s+1)/2** は bit-split 専用。**cuBLAS 内部**はさらに最適化されている可能性。 |
| **`get_bits_per_int8(K)` 未採用** | DG では K=Nq≤256 固定。**127 量子化で INT32 安全**（|Σ|≈4×10⁶ ≪ 2×10⁹）。 | K≈8000 の一般問題では ozIMMU が **K に応じ bit 幅を縮め** INT32 積を守る。 |
| **FP64 直接加算再構成**（ozIMMU の指数シフト累積でない） | CRT 不要の Scheme I そのもの。**assembly 前の residue** として既存経路と接続。 | ozIMMU の shift 累積は **スライス index ごとの厳密な位相**を保つ。 |
| **明示 cublasGemmEx INT8** | 呼び出し回数・帯域が **計測可能**。OZAKI2 と同じ `cuda_cublas_gemm.cu` ラッパ。 | cuBLAS EAGER emulation は **130× 遅**（[`cublas_emulation_survey.md`](cublas_emulation_survey.md)）だが **ビット一致に近い**。比較用には `CublasEmulation` を使う。 |
| **固定 `OzakiSliceCount`** | 再現性のある **A/B 比較**（s を振って誤差–コスト曲線）。 | ozIMMU **`fp64_int8_auto`** / cuBLAS **ADP** は入力に応じた動的精度。 |
| **DG 専用 workspace・NN/TN のみ** | Ne, Nq 既知。**余計な pad / 8 GiB cuBLAS workspace 不要**。 | GEMMul8 `workSize` の 1024 整列・8 GiB 定数は **任意 M,N,K BLAS** 向け。 |
| **B キャッシュはポインタ同一性のみ** | メッシュ固定で **D1D / D1D_tr は不変**。実装が単純。 | 係数が時間変化する一般 BLAS では **内容ハッシュ invalidation** が必要。 |
| **ADP / mantissa loss 自動選択なし** | 性能最適化目的ではなく **Scheme I 経路のモデル化**が目的。 | ozIMMU `auto_mode_select` は **mantissa loss 閾値**で s を最小化。 |

### 2.4 本リポジトリ内ベンチとの関係

| ベンチ | 役割 | 本実装との差 |
|---|---|---|
| `CublasEmulation` | cuBLAS **内蔵** Ozaki I（EAGER） | 本体統合済み・**別フラグ**。速度は使えないが精度参照。 |
| `bench_ozaki2/ozaki1_crt_test.cu` | 合成データで native DGEMM 比較 | 本番 `ozaki1_dgemm*` API の **回帰テスト**。 |
| `bench_ozaki2/mini_ozaki2_all.cu`（Ozaki I API 差替） | 単体 GEMM 誤差比較 | OZAKI1/2 で **同一 INT8 核**のとき誤差一致を確認済み。 |

---

## 3. アルゴリズム（`cuda_ozaki1_gemm.cu`）

1. **B の分解**（`decompose_b_nn`）: 列 max / 127 で量子化 → 残差が閾値超なら次スライス（最大 `OzakiSliceCount`）。
2. **A の分解**（`decompose_a_tn`）: 行 max、pack は TN 形（OZAKI2 と同じ cuBLAS 呼び出し `transa=T`）。
3. **スライスペアループ**（`run_slice_pairs`）: 各 `(i,j)` で INT8 GEMM → `ozaki1_recon_*` が
   `C += scale_a[i,row] × prod × scale_b[j,col]`（最初のペアのみ上書き、以降加算）。
4. **同期**: 各 `ozaki1_dgemm*` 末尾で `dg_cuda_stream` を synchronize。

INT32 飽和: 内積次元 K = Nq ≤ 256 なので 1 回の INT8 GEMM あたり |Σ| ≤ K×127² ≈ 4.1×10⁶
（INT32 上限 2.1×10⁹ に対し余裕。p=255 でも K=256 のため安全）。

---

## 4. 実装時に修正した欠陥

### 4.1 `scale_a` バッファ不足（z 方向 GEMM）

**症状**: p=7 で `dqdt` が NaN / 10⁴ オーダーの外れ値、`OzakiSliceCount=2` では
`ozaki1_dgemm z-deriv` で `CUDA_ERROR_ILLEGAL_ADDRESS`。

**原因**: `scale_a` を `slices × Nq × max_batch` で確保していたが、z 導関数は
`m = nq²` 行（p=7 では 64 行）。batched row-max が `scale[b×m + row]` に書き、
Nq=8 分の領域を **8 倍オーバーラン**。

**修正**: `max_m = Nq²` を workspace に持ち、`scale_a` とスライス間オフセットを
`slices × max_m × max_batch` に拡張。OZAKI2 も同型のインデックスを使うが、
OZAKI2 は slice あたり scale を上書き再利用するため顕在化しにくかった。

### 4.2 ゼロ行のスケール

**症状**: 上記修正前は定数速度 p=7 で q が ±1.8×10³⁰⁸ に発散。

**原因**: max=0 の行に `scale=1.0` を置き、`scales_need_next_slice` が
「127 > 1.0」で真となり、**無意味な追加スライス**（最大 s² ペア）を量産。

**修正**: ゼロ行は `scale=0`、pack 時は `denom>0` のときのみ除算。

---

## 5. OZAKI2 との比較（設計・トレードオフ）

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

## 6. 数値検証

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

### 6.2 変動係数・1000 step 長時間 run（p=7, Ne=32³）

login ノード、`SCALE_DG_VARYING_COEFF=1`、`dt=10⁻⁵`、`nstep=1000`、`WarmupStep=1`
（測定 999 step）、`s=8`。入力は
[`namelists/val_p7_gemm_ozaki1_ne32_n1000.conf`](../namelists/val_p7_gemm_ozaki1_ne32_n1000.conf)
（比較 native は [`namelists/val_p7_gemm_ne32_n1000.conf`](../namelists/val_p7_gemm_ne32_n1000.conf)）。

| 観測 | OZAKI1 | native `CUDAFORTRAN_GEMM` |
|---|---|---|
| step 1000 の q min / max | −1.007 / **+1.007** | −1.007 / **+1.007** |
| 時間積分として | **安定**（|q| ≈ 1 付近） | **同様**（Ozaki 固有の発散なし） |

**解釈**: `dt=10⁻³` では step 200 付近から |q| が指数関数的に増え、step 300 前後で
両経路とも ±1.80×10³⁰⁸ に飽和した（Ozaki 固有ではない）。本節の slice 統計・性能は
**積分が安定する `dt=10⁻⁵`** で再取得した。変動係数下でも |q| が O(1) に留まるため、
実効 pairs_sum は定数速度 1 step（30）からわずかに増える **36/step** にとどまる
（§7.2）。旧 `dt=10⁻³` run の mean pairs_sum=78.7 は **発散に伴う flux 拡大**の
アーティファクトと見なす。

---

## 7. 性能（参考、login ノード・nstep=1）

device イベント行（`CUDA device Ozaki-I GEMM` / `GEMM tendency`）を 1 step 分として記録。
**login ノード共有 GPU のためばらつきあり**。傾向比較用。

| 経路 | p=7 Ne=32³ | p=255 Ne=1 |
|---|---:|---:|
| `CUDAFORTRAN_GEMM` | 39.6 ms | 24.0 ms |
| `CUDAFORTRAN_GEMM_OZAKI1`（s=8） | **127 ms（≈3.2×）** | 15.1 ms |
| `CUDAFORTRAN_GEMM_OZAKI2`（s=14） | 73 ms（≈1.8×） | 48.4 ms |

p=7 では **s² まで増える INT8 GEMM** のため OZAKI2 より遅いのが典型。p=255 Ne=1 の 1 回測定では
OZAKI1 が native より速く見えたが、スライスが早期打ち切りされた場合の **ノイズ**と見なし、
本番最速経路としては採用しない。実効 s_a/s_b は
`SCALE_DG_OZAKI1_SLICE_STATS=1` で run 終了時に min/max/mean と step あたり
`s_a*s_b` 合計（INT8 GEMM 本数の proxy）が出力される。

### 7.2 変動係数・1000 step（§6.2 同一 run、`dt=10⁻⁵`）

環境: `SCALE_DG_VARYING_COEFF=1`、`SCALE_DG_OZAKI1_SLICE_STATS=1`、測定 999 step。

**性能**（Cal_tend 合計 / 999 step）:

| 経路 | Cal_tend / step | CUDA device GEMM / step | native 比 |
|---|---:|---:|---:|
| `CUDAFORTRAN_GEMM` | **5.07 ms** | 4.86 ms | 1.00 |
| `CUDAFORTRAN_GEMM_OZAKI1` | **60.9 ms** | 60.7 ms | **≈12.0×** |

定数速度 1 step（§7）の **≈3.2×** より悪化するが、旧 `dt=10⁻³` 発散 run の
**≈16.9×** よりは軽い。差の主因は **pairs_sum**（下表）: 安定積分では
36/step と定数速度 30 に近く、発散 run の mean 78.7 は flux 飽和の副産物。

**実効スライス**（999 step × 9 GEMM 呼び出し = 8991 サンプル）:

| 量 | min | max | mean |
|---|---:|---:|---:|
| s_a / 呼び出し | 2 | 2 | 2.00 |
| s_b / 呼び出し | 2 | 2 | 2.00 |
| s_a×s_b / 呼び出し | 4 | 4 | 4.00 |
| **s_a×s_b 合計 / step** | **36** | **36** | **36** |

参照（定数速度・1 step）: pairs_sum/step = **30**（p=7 Ne=32³）、**15**（p=255 Ne=1）。
変動係数・安定積分では **36**（定数 p=7 の **1.2 倍**）— 係数変動だけでは s_a=s_b=2
が全 step で固定。§7 の p=255「15 ms」は pairs_sum=15 の best case。
旧 `dt=10⁻³` run（mean 78.7、max 304）は積分発散時の flux 拡大を反映しており、
本番性能見積もりには使わない。

---

## 8. メモリ（workspace 概算）

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
| `scale_a` | s × **Nq²** × max_batch × 8 B（§4.1 修正後） |
| `scale_b` | s × Np × Ne × 8 B（非 batched B 最大列数は nq²×Ne だが batched 小 B は先頭のみ使用） |
| `prod` | Np × Ne × 4 B |
| `res_a`, `res_b` | Np × Ne × 8 B |

p=255 Ne=1、s=8 では `scale_a` だけ **約 1.0 GiB**（Nq²×256×8×8）。OZAKI2 の
residue 940 MB と同オーダーだが、**s² 回 GEMM** の方が時間側のペナルティが大きい。

---

## 9. 統合上の制約

- **CUDA Graph 非対応**: ホスト側 slice 判定・可変 s_a/s_b・stream sync のため
  `advect3d_eq_graph_supported` が false（OZAKI2 と同様）。
- **p=255**: `PolyOrder=255` ゲートに OZAKI1 を追加済み。
- **Ne 汎用**: x/y/z の strided batched は OZAKI2 と同じ stride 契約（`strideB=0` で
  共有 B）。`Ne>1` smoke は p=7 で実施。

---

## 10. 再現方法

```bash
module load nvhpc
export CUTLASS_HOME=/path/to/third_party/cutlass
cd scale-dg-kernel-extraction-ozaki
make clean && make CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100

cd bench_ozaki2
nvcc -O3 -arch=sm_100 -I.. ../cuda_ozaki1_gemm.cu ../cuda_cublas_gemm.cu \
  ozaki1_crt_test.cu -lcublas -o ozaki1_crt_test
./ozaki1_crt_test

# DG smoke
./scale-dg_extraction namelists/val_p7_gemm_ozaki1_ne32.conf
./scale-dg_extraction namelists/val_p255_gemm_ozaki1.conf

# 実効スライス統計（warmup 後の step を集約、run 終了時に表示）
export SCALE_DG_OZAKI1_SLICE_STATS=1
./scale-dg_extraction namelists/val_p7_gemm_ozaki1_ne32.conf
# 1 step ごとの内訳: SCALE_DG_OZAKI1_SLICE_STATS_VERBOSE=1 も併用

# 変動係数・1000 step（dt=10⁻⁵、slice 統計 + 性能）
export SCALE_DG_VARYING_COEFF=1
export SCALE_DG_OZAKI1_SLICE_STATS=1
./scale-dg_extraction namelists/val_p7_gemm_ozaki1_ne32_n1000.conf   # OZAKI1 約 61 s
# native 比較（同条件・約 5 s）
./scale-dg_extraction namelists/val_p7_gemm_ne32_n1000.conf

# dqdt 比較（p=7）
./scale-dg_extraction namelists/val_p7_gemm.conf
export SCALE_DG_DUMP_DQDT=dqdt_ref.txt
./scale-dg_extraction namelists/val_p7_gemm.conf
export SCALE_DG_DUMP_DQDT=dqdt_ozaki1.txt
./scale-dg_extraction namelists/val_p7_gemm_ozaki1_ne32.conf
# paste + awk で max abs diff（約 21）
```

非 CUDA ビルド: `make clean && make` — stub に OZAKI1 シンボルあり。

---

## 11. まとめ

- **Ozaki Scheme I** を `CUDAFORTRAN_GEMM_OZAKI1` として volume GEMM 3 本に統合。
  CRT なし・A/B 両スライス・FP64 直接加算が Scheme II との本質差。
- **参照比較**: GEMMul8 の Ozaki I は **cuBLAS 内蔵**、オープン参照は **ozIMMU**。
  残差 max/127 分解・s_a×s_b ペア・固定 s は **DG 点ごと flux** と **再現可能な計測**に特化。
- **取り込み**: ゼロ行スケール、`scale_a` 拡張、**D1D_tr B 分解キャッシュ**、`ozaki1_crt_test`。
- **性能結論**: 定数速度 p=7 では native の **約 3 倍**（§7）。**変動係数 1000 step**
  （`dt=10⁻⁵`・安定積分）では pairs_sum=36 により **約 12 倍**（§7.2）。
  `dt=10⁻³` 発散 run の ≈17 倍は flux 飽和のアーティファクト。p=255 1 step の
  「native より速い」は pairs_sum=15 の外れ値（§7）。
- **今後**: ozIMMU 式 **mantissa bit split** の ablation、変動係数 p=255 全点比較、
  B キャッシュ効果の nsys 計測は任意。

関連: [`ozaki2_implementation_report.md`](ozaki2_implementation_report.md)、
[`ozaki2_survey_2504.08009.md`](ozaki2_survey_2504.08009.md)、
[`cublas_emulation_survey.md`](cublas_emulation_survey.md)、
[`AGENTS.md`](../AGENTS.md)。
