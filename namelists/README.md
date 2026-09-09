# Namelists

Committed inputs are the samples needed to run the code and to reproduce
the published timing mesh. Experiment-only copies (`_ncu`, `_n400`, extra
`Ne`, load-audit) stay out of git: copy a `perf_` file and change `nstep`
or `DqdtKernel_Type`.

**経路別の `perf_` / `val_` はコミットされている。** 上の「コピーして
`DqdtKernel_Type` を変えろ」は *experiment-only* のコピー（短い `nstep`、
別 `Ne`、ロード監査）に対する規則であって、published mesh 上の各経路の
入力に対する規則ではない。published mesh の 1 セル
（1 つの次数 × 1 つの `DqdtKernel_Type`）は測定の再現に必要なので、
`perf_p255_gemm_ozaki1.conf` のような経路別ファイルもコミットされている。
現物は下の「Committed files」の網羅表のとおり 123 本。

どちらが正典かは**実行のしかたで決まる**:

| 実行 | 使う入力 | 出所 |
|---|---|---|
| クロスマシン計測ハーネス（`jobs/tsubame_paths.sh` と `rikyu_/wisteria_` ラッパ） | **その場生成**。コミット済みの経路別 conf は読まない | `make_config()` が `namelists/perf_p{order}_gemm.conf` **1 本だけ**をテンプレートにし、`DqdtKernel_Type`（と要求時 `CutlassMmaShape`）を `sed` で差し替えて `$OUTDIR/config/` に書き出す。validation 用も同じテンプレートから作り、`make_config()` の**呼び出し側**が続けて `nstep = 1` と `output_interval = 1` に落とす |
| 手動実行・単発の再現・`ncu`/`nsys` の元ファイル | **コミット済みの `perf_*` / `val_*` をそのまま**使う | 下の表 |

つまりハーネスにとって必須なのは各次数の `perf_p{order}_gemm.conf`
（テンプレート、`p=7,15,31,63,127,255,511,575,767` の 9 本）だけで、
残りの経路別 conf は手動実行と、レポートが「この入力で測った」と名指し
できる対象としてコミットされている。**両者は同じメッシュを指しているが、
同期は自動ではない**: `perf_p{order}_gemm.conf` の `Ne`/`dt`/`nstep` を
変えるとハーネスの全経路が動き、経路別 conf は取り残される。メッシュを
変えるときはその次数の全ファイルを一緒に直すこと。

Run from the repository root:

```bash
./scale-dg_extraction namelists/demo_p7_openacc_split_n1000.conf
./scale-dg_extraction namelists/perf_p63_fused_tc.conf
```

Nsight Compute needs a short `nstep` (typically 4). Do not keep a parallel
`_ncu` file; copy the perf namelist and shorten it.

## Naming

```
{purpose}_p{order}_{kernel}[_{qualifiers}].conf
```

| Field | Values |
|---|---|
| `purpose` | `demo` (tiny smoke), `perf` (timing mesh), `val` (point-varying coefficient check), `acc` (time-integrated accuracy, `SCALE_DG_ACCURACY=1`) |
| `order` | polynomial order (`7`, `15`, …, `1023`) |
| `kernel` | `openacc_split`, `split`, `fused`, `fused_tc`, `fused_dfma`, `gemm`, `gemm_cute`, `gemm_fused`, `gemm_ozaki1`, `gemm_ozaki2` |

Qualifiers only when they are not the default: `emu` (`CublasEmulation`), `ne{N}`, `n{N}`.

## Committed files

コミット済み `.conf` は **123 本**（`git ls-files namelists/ | grep -c '\.conf$'`）。
1 行 1 ファイルだと 123 行になり読めないので、**次数を行・`kernel` を列**に
した網羅表を正典とし、そのあとに個別の役割メモを置く。

### 網羅表: order × kernel

セルはその組み合わせで実在する `purpose`。`[...]` は既定でない
qualifier（`emu` = `CublasEmulation = .true.`、`ne{N}`、`n{N}`）。
`-` はファイルが無い組み合わせ。
ファイル名はセルから機械的に復元できる:
`{purpose}_p{order}_{kernel}[_{qualifiers}].conf`。

| p | openacc_split | split | fused | fused_tc | fused_dfma | gemm | gemm_cute | gemm_fused | gemm_ozaki1 | gemm_ozaki2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `7` | demo[n1000], perf[n1000] | val | perf, val | perf, val | perf | acc, perf, perf[emu,n1], val, val[emu], val[ne32,n1000] | perf | perf, val | perf, val[ne32], val[ne32,n1000] | perf, val[ne32] |
| `15` | - | val | perf, val | perf | perf | perf | perf | perf, val | perf | perf |
| `31` | - | val | perf, val | perf | perf | perf | perf | perf, val | perf | perf |
| `63` | - | val | perf, val | perf, val | perf, val | acc, perf | perf | perf | perf | perf |
| `127` | - | val | perf, val | perf, val | perf | perf | perf | perf | perf | perf |
| `255` | - | - | perf, val | perf, val | perf | acc, perf, val, val[emu] | perf, val | perf, val | perf, val | perf, val |
| `511` | - | - | - | perf, val | perf, val | perf, val, val[emu,n20] | perf | perf, val | perf | perf |
| `575` | - | - | - | - | - | perf, val, val[emu,n50] | perf | perf, val | perf | perf |
| `767` | - | - | - | val | perf, val | perf, val, val[emu,n30] | perf, val | perf, val | perf | perf |
| `1023` | - | - | - | - | - | perf, val, val[emu,n15] | perf | perf, val | - | - |

内訳: `perf` 71、`val` 48、`acc` 3、`demo` 1。全 123 本について
ファイル本文の `PolyOrder` / `DqdtKernel_Type` / `CublasEmulation` が
ファイル名と一致することを確認済み（不整合ゼロ）。

穴（実在しない組み合わせ）のうち意図があるもの:

- p=511 以上に `fused`（CUDA-core 融合）が無い。p≥511 では CC 融合が
  動かない／勝たない（`reports/p511_gap_study.md` §14.6, §15.7）。
  `jobs/tsubame_paths.sh` の `paths_for_order()` も p=511 以上で
  `CUDAFORTRAN_FUSED` を外している。
- p=575 と p=1023 には `fused*` 系が一切無い。GEMM 系のみの次数。
- `gemm_ozaki{1,2}` は p=1023 のみ欠。
- `perf_p767_fused_tc.conf` が無い（`val_p767_fused_tc.conf` と
  `perf_p767_fused_dfma.conf` はある）。dfma 側だけ perf があり TC 側が欠ける。
  **対応不明の欠落**であって、意図された穴とは確認できていない。
- `val_p{15,31,63,127}_gemm_cute.conf` が無い（`val` の CUTE は p=255/767 のみ）。

### メッシュ（`purpose` ごとの `Ne`/`nstep`）

`perf` は published timing mesh。`NeX = NeY = NeZ`。

| p | 7 | 15 | 31 | 63 | 127 | 255 | 511 | 575 | 767 | 1023 |
|---|---|---|---|---|---|---|---|---|---|---|
| `perf` `Ne` | 32 | 16 | 8 | 4 | 2 | 1 | 1 | 1 | 1 | 1 |
| `perf` `nstep` | 20 | 20 | 200 | 20 | 100 | 20 | 20 | 50 | 15 | 10 |
| `val` `Ne` | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 | 1 | 1 |

`val` は `nstep = 1`（最初の tendency 評価で `dqdt` を比べるため）。
例外は `val_p7_gemm_ne32_n1000.conf` と `val_p7_gemm_ozaki1_ne32_n1000.conf`
（`Ne=32`, `nstep=1000`、時間積分後まで走らせる版）、および
高次の `val_*_gemm_emu_n{15,20,30,50}.conf`（emulation は 1 ステップでは
差が出ないので `nstep` を伸ばしてある）。
`acc_p{7,63,255}_gemm.conf` の `nstep` は下の役割メモ参照。
`perf_p7_gemm_emu_n1.conf` は `nstep = 1`。

### 個別の役割メモ

網羅表で役割が読み取れないファイルの補足。ここに無いファイルは
表のセルがそのまま役割（published mesh のその経路の入力）。

| File | Role |
|---|---|
| `demo_p7_openacc_split_n1000.conf` | README smoke (`Ne=4`, p=7) |
| `perf_p7_openacc_split_n1000.conf` | p=7 GPU benchmark (`Ne=32`) |
| `perf_p7_fused.conf` | published p=7 mesh, CUDA-core fused (`Ne=32`, `nstep=20`) |
| `perf_p7_fused_tc.conf` | published p=7 mesh (`Ne=32`, `nstep=20`) |
| `perf_p7_gemm_fused.conf` | published p=7 mesh, `CUDAFORTRAN_GEMM_FUSED` (`Ne=32`, `nstep=20`) |
| `perf_p15_fused_tc.conf` | published p=15 mesh (`Ne=16`, `nstep=20`) |
| `perf_p15_fused.conf` | p=15 CUDA-core fused (`CUDAFORTRAN_FUSED`) |
| `perf_p15_gemm_fused.conf` | published p=15 mesh, `CUDAFORTRAN_GEMM_FUSED` |
| `perf_p31_fused_tc.conf` | published p=31 mesh (`Ne=8`, `nstep=200`) |
| `perf_p31_gemm_fused.conf` | published p=31 mesh, `CUDAFORTRAN_GEMM_FUSED` |
| `perf_p63_fused_tc.conf` | published p=63 mesh (`Ne=4`, `nstep=20`) |
| `perf_p127_gemm_fused.conf` | published p=127 mesh (`Ne=2`, `nstep=100`) |
| `perf_p127_fused.conf` | same mesh, `CUDAFORTRAN_FUSED` |
| `perf_p127_fused_tc.conf` | same mesh, `CUDAFORTRAN_FUSED_TC` |
| `val_p127_fused_tc.conf` / `val_p127_split.conf` | p=127 point-varying `dqdt` check (`Ne=2`, `nstep=1`) |
| `perf_p255_fused_tc.conf` | published p=255 mesh (`Ne=1`, `nstep=20`) |
| `perf_p255_fused.conf` | same mesh, `CUDAFORTRAN_FUSED`（CUDA-core 融合） |
| `perf_p575_gemm_fused.conf` | published p=575 mesh (`Ne=1`, `nstep=50`) |
| `perf_p{767,1023}_gemm_fused.conf` | same for the other `Nq>=512` degrees (`nstep` 15 / 10) |
| `perf_p767_gemm_cute.conf` / `val_p767_gemm_cute.conf` | p=767 の未融合対照 `CUDAFORTRAN_GEMM_CUTE`（同一メッシュ・同一 `nstep`） |
| `perf_p7_gemm_emu_n1.conf` | cuBLAS emulation timing (do not use `nstep=1000`) |
| `val_p7_*` / `val_p7_split.conf` / `val_p15_fused.conf` / `val_p15_split.conf` / `val_p15_gemm_fused.conf` / `val_p31_split.conf` / `val_p31_gemm_fused.conf` / `val_p63_split.conf` / `val_p63_fused_tc.conf` / `val_p127_fused.conf` / `val_p127_split.conf` / `val_p255_*` | numerical checks (`SCALE_DG_VARYING_COEFF=1`) |
| `acc_p{7,63,255}_gemm.conf` | 時間積分後の精度検証。定数速度・周期領域なので厳密解は初期条件の平行移動で、`SCALE_DG_ACCURACY=1` を付けると L2 / L∞ 誤差と質量ドリフトを出す。`DqdtKernel_Type` を差し替えて全経路を回す用途（`reports/accuracy_validation.md`）。`nstep` は t_end が p=7 / 63 で 0.01、p=255 で 0.001 になるよう選んである |
| `perf_p511_gemm.conf` | p=511 published mesh, `CUDAFORTRAN_GEMM` (`Ne=1`, `nstep=20`) |
| `perf_p511_gemm_fused.conf` | p=511 published mesh, `CUDAFORTRAN_GEMM_FUSED` (`Ne=1`, `nstep=20`) |
| `val_p{511,575,767,1023}_gemm*.conf` | high-order GEMM / GEMM_FUSED / emu |
| `perf_p511_gemm_cute.conf` | p=511 の未融合対照 `CUDAFORTRAN_GEMM_CUTE`（同一メッシュ・同一 `nstep`。`p511_gap_study.md` §15） |
| `perf_p511_fused_tc.conf` / `val_p511_fused_tc.conf` | p=511 の `CUDAFORTRAN_FUSED_TC`（同一メッシュ・同一 `nstep`。`p511_gap_study.md` §15） |
| `perf_p511_fused_dfma.conf` / `val_p511_fused_dfma.conf` | p=511 の `CUDAFORTRAN_FUSED_DFMA`（iso-schedule 対照。同上） |

## 同じメッシュで別の経路を走らせる

Another path on the same mesh: copy the `perf_` file and change `DqdtKernel_Type` only. Do not change `NeX/NeY/NeZ`, `PolyOrder`, `dt`, or `nstep` when comparing kernels.

網羅表にセルがある組み合わせなら、コピーではなく**その committed ファイルを
そのまま**使う（それがコミットされている理由）。セルが `-` のとき、
あるいは `_ncu` のような experiment-only の変種が要るときにコピーする。
コピー先はコミットしない。

計測ハーネスから走らせるときはコピーもしない: `jobs/tsubame_paths.sh` の
`make_config()` が `perf_p{order}_gemm.conf` から生成する（冒頭の表）。
したがって **`perf_p{order}_gemm.conf` は「`CUDAFORTRAN_GEMM` の入力」で
あると同時にその次数のテンプレートでもある**。この 9 本
（p=7, 15, 31, 63, 127, 255, 511, 575, 767）は、経路を追加・削除しても
消してはならない。p=1023 の `perf_p1023_gemm.conf` は
`paths_for_order()` が p=1023 を返さないのでハーネスからは使われないが、
手動実行の入力として同じ形で置いてある。
