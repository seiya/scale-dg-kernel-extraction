# 旧 namelist 名 → 現行名の対応表

## 0. このファイルの役目

`reports/` 配下のレポートは、測定当時に使った namelist をそのファイル名で引用する。
`reports/README.md`:8-10 のとおり、**本文の旧名は書き換えない**（測定済みの記録は
証拠であって、現在のツリーに対する主張ではない）。その代わりに、旧名から現行名を
引くための受け皿がこのファイルである。同じ役目の job スクリプト側は
`jobs/MAPPING.md`（`jobs/` は allowlist 追跡なので、そのファイルは
`.gitignore` の allowlist に加えてある）。

### いつ名前が変わったか

- `9dfea48` "Keep only sample and published-mesh namelists under namelists/."
  (2026-08-30) が唯一の改名コミット。ルート直下にあった `conf_perf_*.conf` /
  `input_*.conf` をコミット対象と一時ファイルに切り分け、残すものだけを
  `namelists/{purpose}_p{order}_{kernel}[_{qualifiers}].conf` に改名した。
  同じコミットで `namelists/README.md` が新設され、`reports/README.md` に
  「旧名はそのまま残す」方針の 5 行が入っている。
- `9dfea48..HEAD` に `namelists/` 内の改名は無い（`git log --diff-filter=R`）。
  以後に増えたファイルはすべて現行規則で新規追加されている。

### 旧名がリポジトリに残っていない理由は 2 通りある

1. `9dfea48` で改名または削除された（mainline の履歴 `9dfea48^` に実体がある）。
2. **そもそも mainline にコミットされたことがない**作業ツリー上の一時ファイル。
   これらの一部は、当時の作業ツリーを丸ごと取り込んだ Cursor の作業ブランチ
   `cursor/cloud-agent-1787998302748-eyapx` の先頭コミット `6466dc0`
   (2026-08-29) にだけ実体がある（`6466dc0` は HEAD の祖先ではない）。
   本表では実体の在処を「`6466dc0`（cursor ブランチのみ）」と記す。

対応の裏取りは、旧ファイルの `PolyOrder` / `DqdtKernel_Type` / `NeX,NeY,NeZ` /
`nstep` / `CublasEmulation` を現行ファイルのそれと突き合わせて行った。
**内容で裏が取れないものは §3 に置き、対応を推測で埋めていない。**

## 1. 旧命名規則の読み方（参考）

- `conf_perf_p{order}[_{kernel}][_ncu].conf` — 性能測定用。`kernel` 省略は
  「当時その次数で主に測っていた経路」であって既定値の意味ではない
  （実際 `conf_perf_p63.conf` は `CUDAFORTRAN_GEMM`、`conf_perf_p255.conf` は
  `CUDAFORTRAN_GEMM_FUSED`）。`_tc` = `fused_tc`、`_gf` = `gemm_fused`、
  `_ncu` = `nstep` を 4〜5 に落とした ncu 用コピー。
- `input_p{order}_val[_{kernel}][_emu|_ref].conf` — 検証用。`_ref` = 参照実装
  （`*_SPLIT`）、`_emu` = `CublasEmulation=.true.`。
- `input_large*.conf` / `input.conf` — 次数を名前に持たない最初期の名前。
  実体の分かる 2 本（`input.conf`, `input_large.conf`）は §2.2。

## 2. 対応表（確定）

### 2.1 旧ファイルの内容と現行ファイルの内容が一致するもの

`(order, kernel)` は `PolyOrder` と `DqdtKernel_Type` の実測値。
「実体」は旧ファイルが残っている場所。

| 旧名 | 現行名 | (order, kernel) | 引用しているレポート | 実体 |
|---|---|---|---|---|
| `conf_perf_p7.conf` | `namelists/perf_p7_fused_tc.conf` | (7, `CUDAFORTRAN_FUSED_TC`) | overall_summary_report, p15_gap_study, tc_paper_survey_2407.09621, README | `6466dc0` |
| `conf_perf_p15.conf` | `namelists/perf_p15_fused.conf` | (15, `CUDAFORTRAN_FUSED`) | overall_summary_report, p15_gap_study | `6466dc0` |
| `conf_perf_p15_tc.conf` | `namelists/perf_p15_fused_tc.conf` | (15, `CUDAFORTRAN_FUSED_TC`) | p15_gap_study | `6466dc0` |
| `conf_perf_p31_fused.conf` | `namelists/perf_p31_fused.conf` | (31, `CUDAFORTRAN_FUSED`) | overall_summary_report, p31_gap_study | `6466dc0` |
| `conf_perf_p31_tc.conf` | `namelists/perf_p31_fused_tc.conf` | (31, `CUDAFORTRAN_FUSED_TC`) | p31_gap_study, p63_gap_study | `9dfea48^`（同コミットで `git mv` 相当、内容 100% 一致） |
| `conf_perf_p63.conf` | `namelists/perf_p63_gemm.conf` | (63, `CUDAFORTRAN_GEMM`) | p63_gap_study | `6466dc0` |
| `conf_perf_p63_fused.conf` | `namelists/perf_p63_fused.conf` | (63, `CUDAFORTRAN_FUSED`) | overall_summary_report, p63_gap_study, README | `9dfea48^` |
| `conf_perf_p63_gemm_fused.conf` | `namelists/perf_p63_gemm_fused.conf` | (63, `CUDAFORTRAN_GEMM_FUSED`) | p63_gap_study, README | `6466dc0` |
| `conf_perf_p63_tc.conf` | `namelists/perf_p63_fused_tc.conf` | (63, `CUDAFORTRAN_FUSED_TC`) | p63_gap_study, README | `6466dc0` |
| `conf_perf_p127_fused.conf` | `namelists/perf_p127_fused.conf` | (127, `CUDAFORTRAN_FUSED`) | overall_summary_report, p127_gap_study | `6466dc0` |
| `conf_perf_p127_gemm_fused.conf` | `namelists/perf_p127_gemm_fused.conf` | (127, `CUDAFORTRAN_GEMM_FUSED`) | p127_gap_study | `6466dc0` |
| `conf_perf_p127_tc.conf` | `namelists/perf_p127_fused_tc.conf` | (127, `CUDAFORTRAN_FUSED_TC`) | p127_gap_study | `6466dc0` |
| `conf_perf_p255.conf` | `namelists/perf_p255_gemm_fused.conf` | (255, `CUDAFORTRAN_GEMM_FUSED`) | overall_summary_report, p31_gap_study, p255_gap_study, README | `6466dc0` |
| `conf_perf_p255_tc.conf` | `namelists/perf_p255_fused_tc.conf` | (255, `CUDAFORTRAN_FUSED_TC`) | overall_summary_report, p255_gap_study | `6466dc0` |
| `input_p7_val_gemm.conf` | `namelists/val_p7_gemm.conf` | (7, `CUDAFORTRAN_GEMM`) | cublas_emulation_survey | `9dfea48^` |
| `input_p7_val_gemm_emu.conf` | `namelists/val_p7_gemm_emu.conf` | (7, `CUDAFORTRAN_GEMM` + `CublasEmulation=.true.`) | cublas_emulation_survey | `9dfea48^` |
| `input_p7_val_gemm_ozaki1.conf` | `namelists/val_p7_gemm_ozaki1_ne32.conf` | (7, `CUDAFORTRAN_GEMM_OZAKI1`, Ne=32³) | ozaki1_implementation_report | `9dfea48^` |
| `input_p7_val_gemm_ozaki1_1000.conf` | `namelists/val_p7_gemm_ozaki1_ne32_n1000.conf` | (7, `CUDAFORTRAN_GEMM_OZAKI1`, Ne=32³, nstep=1000) | ozaki1_implementation_report | `9dfea48^` |
| `input_p7_val_gemm_ozaki2.conf` | `namelists/val_p7_gemm_ozaki2_ne32.conf` | (7, `CUDAFORTRAN_GEMM_OZAKI2`, Ne=32³) | ozaki2_implementation_report | `9dfea48^` |
| `input_p7_val_tc.conf` | `namelists/val_p7_fused_tc.conf` | (7, `CUDAFORTRAN_FUSED_TC`) | tc_paper_survey_2407.09621 | `9dfea48^` |
| `input_p7_perf_gemm_emu.conf` | `namelists/perf_p7_gemm_emu_n1.conf` | (7, `CUDAFORTRAN_GEMM` + emu, Ne=32³, nstep=1) | cublas_emulation_survey, ozaki1_implementation_report | `9dfea48^` |
| `input_p15_val.conf` | `namelists/val_p15_fused.conf` | (15, `CUDAFORTRAN_FUSED`) | p15_gap_study | `6466dc0` |
| `input_p31_val_ref.conf` | `namelists/val_p31_split.conf` | (31, `CUDAFORTRAN_SPLIT`) | p31_gap_study | `9dfea48^` |
| `input_p63_val_ref.conf` | `namelists/val_p63_split.conf` | (63, `CUDAFORTRAN_SPLIT`) | p63_gap_study | `9dfea48^` |
| `input_p255_val_gemm.conf` | `namelists/val_p255_gemm.conf` | (255, `CUDAFORTRAN_GEMM`) | cublas_emulation_survey, p255_gap_study | `9dfea48^` |
| `input_p255_val_gemm_emu.conf` | `namelists/val_p255_gemm_emu.conf` | (255, `CUDAFORTRAN_GEMM` + emu) | cublas_emulation_survey | `9dfea48^` |
| `input_p255_val_gemm_ozaki1.conf` | `namelists/val_p255_gemm_ozaki1.conf` | (255, `CUDAFORTRAN_GEMM_OZAKI1`) | ozaki1_implementation_report | `9dfea48^` |
| `input_p255_val_gemm_ozaki2.conf` | `namelists/val_p255_gemm_ozaki2.conf` | (255, `CUDAFORTRAN_GEMM_OZAKI2`) | ozaki2_implementation_report | `9dfea48^` |
| `input_p255_val_tc.conf` | `namelists/val_p255_fused_tc.conf` | (255, `CUDAFORTRAN_FUSED_TC`) | p255_gap_study | `9dfea48^` |
| `input_p511_val_gemm.conf` | `namelists/val_p511_gemm.conf` | (511, `CUDAFORTRAN_GEMM`) | p511_gap_study | `9dfea48^` |
| `input_p511_val_gemm_emu.conf` | `namelists/val_p511_gemm_emu_n20.conf` | (511, `CUDAFORTRAN_GEMM` + emu, nstep=20) | cublas_emulation_survey, p511_gap_study | `9dfea48^` |
| `input_p511_val_gemm_fused.conf` | `namelists/val_p511_gemm_fused.conf` | (511, `CUDAFORTRAN_GEMM_FUSED`) | p511_gap_study | `9dfea48^` |
| `input_p575_val_gemm.conf` | `namelists/val_p575_gemm.conf` | (575, `CUDAFORTRAN_GEMM`) | p575_gap_study | `9dfea48^` |
| `input_p575_val_gemm_emu.conf` | `namelists/val_p575_gemm_emu_n50.conf` | (575, `CUDAFORTRAN_GEMM` + emu, nstep=50) | cublas_emulation_survey, p575_gap_study | `9dfea48^` |
| `input_p575_val_gemm_fused.conf` | `namelists/val_p575_gemm_fused.conf` | (575, `CUDAFORTRAN_GEMM_FUSED`) | p575_gap_study | `9dfea48^` |
| `input_p767_val_gemm.conf` | `namelists/val_p767_gemm.conf` | (767, `CUDAFORTRAN_GEMM`) | p767_gap_study | `9dfea48^` |
| `input_p767_val_gemm_emu.conf` | `namelists/val_p767_gemm_emu_n30.conf` | (767, `CUDAFORTRAN_GEMM` + emu, nstep=30) | cublas_emulation_survey, p767_gap_study | `9dfea48^` |
| `input_p767_val_gemm_fused.conf` | `namelists/val_p767_gemm_fused.conf` | (767, `CUDAFORTRAN_GEMM_FUSED`) | p767_gap_study | `9dfea48^` |
| `input_p1023_val_gemm.conf` | `namelists/val_p1023_gemm.conf` | (1023, `CUDAFORTRAN_GEMM`) | p1023_gap_study | `9dfea48^` |
| `input_p1023_val_gemm_emu.conf` | `namelists/val_p1023_gemm_emu_n15.conf` | (1023, `CUDAFORTRAN_GEMM` + emu, nstep=15) | cublas_emulation_survey, p1023_gap_study | `9dfea48^` |

### 2.2 レポートが glob でしか触れない 100% 改名

`9dfea48` の改名検出は既定のリネーム上限に引っかかるので、`-l0` が要る:

```
git diff-tree -r -M45% -l0 --name-status 9dfea48^ 9dfea48
```

これで `R100`（内容完全一致）が出る対応のうち、レポート本文が素の名前では
引かず §3.3 の glob（`input_p7_val*.conf`, `input_p255_val_gemm*.conf`,
`input_p*_val*` など）でだけ触れているもの。glob から実体をたどるときはここを見る。

| 旧名 | 現行名 | (order, kernel) |
|---|---|---|
| `input.conf` | `namelists/demo_p7_openacc_split_n1000.conf` | (7, `OPENACC_SPLIT`, Ne=4³, nstep=1000) |
| `input_large.conf` | `namelists/perf_p7_openacc_split_n1000.conf` | (7, `OPENACC_SPLIT`, Ne=32³, nstep=1000) |
| `input_p7_val.conf` | `namelists/val_p7_fused.conf` | (7, `CUDAFORTRAN_FUSED`) |
| `input_p7_val_gemm_1000.conf` | `namelists/val_p7_gemm_ne32_n1000.conf` | (7, `CUDAFORTRAN_GEMM`, Ne=32³, nstep=1000) |
| `input_p255_val.conf` | `namelists/val_p255_fused.conf` | (255, `CUDAFORTRAN_FUSED`) |
| `input_p255_val_gemm_cute.conf` | `namelists/val_p255_gemm_cute.conf` | (255, `CUDAFORTRAN_GEMM_CUTE`) |
| `input_p255_val_gemm_fused.conf` | `namelists/val_p255_gemm_fused.conf` | (255, `CUDAFORTRAN_GEMM_FUSED`) |
| `input_p1023_val_gemm_fused.conf` | `namelists/val_p1023_gemm_fused.conf` | (1023, `CUDAFORTRAN_GEMM_FUSED`) |

### 2.3 同じ検出が出す偽の対応

同じ `-l0` の出力は `conf_perf_p63_fused.conf` → `perf_p63_fused_tc.conf` (62%)、
`input_p63_val_fused.conf` → `perf_p127_gemm_fused.conf` (48%)、
`conf_perf_p31_tc_ncu.conf` → `perf_p15_fused_tc.conf` (45%)、
`input_large_ne1_tc.conf` → `perf_p255_fused_tc.conf` (61%)、
`input_large_fused_tc.conf` → `perf_p7_fused_tc.conf` (51%) のような
**類似度 45〜62% の対応**も出すが、これは削除と追加の取り違えである
（`DqdtKernel_Type` も `PolyOrder` も一致しない）。§2.1 と §2.2 は内容一致だけを採った。

### 2.4 旧ファイルの実体が残っておらず、名前規則だけで一意に決まるもの

旧名が `p{order}` と `{kernel}` を明示しており、現行に同じ (order, kernel) の
`perf_` が 1 本だけ存在するもの。**`NeX/NeY/NeZ` と `nstep` が当時と同じかは
確認できない**（旧ファイルがどのブランチにも無い。`git log --all
--diff-filter=A` で 0 件）。数値を再現するときは現行ファイルのメッシュと
`nstep` を確認すること。

| 旧名 | 現行名 | (order, kernel) | 引用しているレポート |
|---|---|---|---|
| `conf_perf_p575_gemm.conf` | `namelists/perf_p575_gemm.conf` | (575, `CUDAFORTRAN_GEMM`) | p575_gap_study |
| `conf_perf_p575_gemm_fused.conf` | `namelists/perf_p575_gemm_fused.conf` | (575, `CUDAFORTRAN_GEMM_FUSED`) | p575_gap_study |
| `conf_perf_p767_gemm.conf` | `namelists/perf_p767_gemm.conf` | (767, `CUDAFORTRAN_GEMM`) | p767_gap_study |
| `conf_perf_p767_gemm_fused.conf` | `namelists/perf_p767_gemm_fused.conf` | (767, `CUDAFORTRAN_GEMM_FUSED`) | p767_gap_study |
| `conf_perf_p1023_gemm.conf` | `namelists/perf_p1023_gemm.conf` | (1023, `CUDAFORTRAN_GEMM`) | p1023_gap_study |
| `conf_perf_p1023_gemm_fused.conf` | `namelists/perf_p1023_gemm_fused.conf` | (1023, `CUDAFORTRAN_GEMM_FUSED`) | p1023_gap_study |

`/tmp/conf_perf_p511.conf`, `/tmp/conf_perf_p575_gemm*.conf`,
`/tmp/conf_perf_p767_gemm*.conf`, `/tmp/conf_perf_p1023_gemm*.conf`
（p511_gap_study / p575_gap_study / p767_gap_study / p1023_gap_study）は
同名ファイルを `/tmp` に置いて走らせた複製で、対応先は §2.4 の表と同じ。
`/tmp/conf_perf_p511.conf` と `/tmp/conf_perf_p511_fused.conf` は例外で
§3.2 に置く（§2.4 に p=511 の行は無い）。

`/tmp/acc.conf`（accuracy_validation）も `/tmp` 上の一時ファイルだが、こちらは
旧名ではない: 引用元の再現手順が `sed "s/DqdtKernel_Type = .../" \
namelists/acc_p63_gemm.conf > /tmp/acc.conf` と生成の仕方ごと書いているので、
**現行の `namelists/acc_p63_gemm.conf` の `DqdtKernel_Type` 差し替え版**で確定する。
§1 の dt 収束だけは `namelists/acc_p7_gemm.conf` の `dt`/`nstep` を替えたもの。

## 3. 対応が確定できないもの

### 3.0 実行ログからは復元できない（何が残っていて何が残っていないか）

対応の確定に使える一次資料は git の履歴とハーネスの成果物だけで、
**実行ログは使えない**。

- **`slurm-*.out` / SGE / TCS のジョブログには namelist が出ない。**
  `main.f90` が印字するのは `DEVICE_MEMORY_CONFIG`、ステップごとの
  `q` の min/max、末尾の時間レポートだけで、`PolyOrder` も
  `DqdtKernel_Type` も `NeX/NeY/NeZ` も `nstep` も出力されない。
  ログ単独では**次数すら**復元できない。
- **ハーネス経由の測定は、実際に使った conf が現物で残っている。**
  `jobs/tsubame_paths.sh` の `make_config` は生成した conf を
  `$OUTDIR/config/` に書き出すので、ローカルの成果物ツリーに数百本ある。
  ただしこれらは**すべて現行命名の生成物**（`perf_p{order}_{PATH}.conf`）で、
  本表が扱う旧名（`conf_perf_*` / `input_p*_val*`）は 1 本も無い。
  ルート直下の手書き conf を使っていた時代の測定はハーネス経由ではないので、
  §3.1 / §3.2 の穴はこれでは埋まらない。
- したがって、下の 2 表の「実体」欄は git の履歴（`9dfea48^` と、
  HEAD の祖先でない `6466dc0`）にあるものだけである。そこに無いものは
  **どこにも無い**。

### 3.1 ncu 用の短 `nstep` コピー（現行に対応ファイルは存在しない）

`namelists/README.md` は「`_ncu` コピーや一回限りの `Ne`/`nstep` 変種を
`namelists/` に追加するな」と定めている（`AGENTS.md` の Important Files も同じ）。
したがって**これらに対応する現行ファイルは無く、作る対象でもない**。
再現するときは下表の「同設定の現行ファイル」を `jobs/` 側の作業ディレクトリに
コピーし、`nstep` だけ落とす。

| 旧名 | 引用元 | 実体 / 当時の中身 | 同設定の現行ファイル（`nstep` のみ相違） |
|---|---|---|---|
| `conf_perf_p31_fused_ncu.conf` | p31_gap_study | `6466dc0`: (31, `FUSED`), nstep=4 | `perf_p31_fused.conf` (nstep=200) |
| `conf_perf_p63_ncu.conf` | p63_gap_study | `6466dc0`: (63, **`GEMM_FUSED`**), nstep=4 | `perf_p63_gemm_fused.conf` (nstep=20) |
| `conf_perf_p127_fused_ncu.conf` | p127_gap_study | `6466dc0`: (127, `FUSED`), nstep=4 | `perf_p127_fused.conf` (nstep=100) |
| `conf_perf_p127_gf_ncu.conf` | p127_gap_study | `6466dc0`: (127, `GEMM_FUSED`), nstep=4 | `perf_p127_gemm_fused.conf` (nstep=100) |
| `conf_perf_p255_ncu.conf` | p255_gap_study | `6466dc0`: (255, `GEMM_FUSED`), nstep=4 | `perf_p255_gemm_fused.conf` (nstep=20) |
| `conf_perf_p255_tc_ncu.conf` | p255_gap_study | `6466dc0`: (255, `FUSED_TC`), nstep=4 | `perf_p255_fused_tc.conf` (nstep=20) |
| `conf_perf_p127_tc_ncu.conf` | p127_gap_study | **どのブランチにも無い**（`git log --all --diff-filter=A` で 0 件）。名前からは (127, `FUSED_TC`) の ncu 用 | 推定に留まるため確定させない。(127, `FUSED_TC`) は `perf_p127_fused_tc.conf` |
| `output/conf_perf_p255_fused_ncu.conf` | p255_gap_study | `6466dc0`: (255, `FUSED`), nstep=4 | `perf_p255_fused.conf` (nstep=20) |

レポートが引用するのは `output/` 付きの形だけで、素の
`conf_perf_p255_fused_ncu.conf` は `reports/` のどこにも現れず、どのブランチにも
実体が無い。

`conf_perf_p31_tc_ncu.conf` と `conf_perf_p63_fused_ncu.conf` は `9dfea48` で
削除されており（`9dfea48^` に実体あり: それぞれ nstep=5）、レポート本文の
引用としては現れない。

### 3.2 現行の `namelists/` に対応するファイルが無いもの

| 旧名 | 引用元 | なぜ特定できないか |
|---|---|---|
| `conf_perf_p511.conf` | p511_gap_study | 実体がどのブランチにも無く、名前に `kernel` フィールドも無い。当時 p=511 で測っていた経路が `GEMM`/`GEMM_FUSED`/`FUSED_TC` のどれかを名前からは決められない。`conf_perf_p63.conf` が `GEMM`、`conf_perf_p255.conf` が `GEMM_FUSED` だったように、無印の指す経路は次数ごとに違う |
| `conf_perf_p511_fused.conf` | p511_gap_study | 実体が無い。(511, `CUDAFORTRAN_FUSED`) は読めるが、現行 `namelists/` に p=511 の `fused` は存在しない（`perf_p511_fused.conf` は無い）。p=511 で CC fused を開いた測定は `b494a42` だが namelist はコミットされていない |
| `input_p15_val_tc.conf` | p15_gap_study | 実体は `6466dc0` にあり (15, `CUDAFORTRAN_FUSED_TC`, Ne=2³, nstep=1) と読めるが、現行 `namelists/` に `val_p15_fused_tc.conf` は無い（p=15 の `fused_tc` は `perf_` のみ） |
| `input_p127_val_ref.conf` | p127_gap_study | 実体は `6466dc0` にあり、`DqdtKernel_Type="OPENACC_SPLIT"`。現行の p=127 参照は `val_p127_split.conf` = `CUDAFORTRAN_SPLIT` で**経路が違う**。役割は同じ（point-varying 係数の参照実装）だが同一ファイルの改名ではない |
| `tmp_ozaki_p15_*.conf` | ozaki 系 | glob 表記の一時ファイル。実体が無く、`*` が何本あったかも復元できない |
| `work/conf/prof_p{7,31,63,127}_{gemm,gemm_cute,gemm_fused}.conf` ほか `work/conf/prof_*.conf` | — | `work/conf/` 系はコミットされたことがない別系統（`prof_p{order}_{kernel}.conf`）。(order, kernel) は名前から読めるが、メッシュ・`nstep` が現行 `perf_` と同じである保証が無い |
| `p511_{GEMM,GEMM_CUTE}_{8x8x4,16x8x4}_{time,ncu}.conf` | — | `jobs/h100_p511_gemm_subtraction.sh` が `namelists/perf_p511_gemm.conf` から `$OUTDIR/config/` にその場生成する（`DqdtKernel_Type` と `CutlassMmaShape` を `sed` で差し替え、`_ncu` 側はさらに `nstep`/`output_interval` を `NSTEP_NCU` に落とす）。形状は namelist の `CutlassMmaShape` フィールドそのものだが、既定でない形状は experiment-only なので `namelists/` に対応物を持たない設計 |
| `/tmp/p7gf_ncu_${SLURM_JOB_ID}.conf` | p7_gemm_fused | ジョブ ID で名前を作る `ncu` 用一時 conf。中身は実行ごとに違い、生成元のジョブスクリプトも未追跡 |
| `jobs/perf_p511_gemm_ozaki{1,2}_{FIXED,ADP}.conf` | — | `jobs/` 直下の未追跡の実験用 conf。`_FIXED` / `_ADP` の 4 本とも現在も未追跡で存在する。いずれも `namelists/` に対応物を持たない設計 |

### 3.3 glob 形の引用

レポート本文には `conf_perf_p63*.conf`, `conf_perf_p31*.conf`,
`conf_perf_p127*.conf`, `conf_perf_p255*.conf`, `conf_perf_*`,
`input_p7_val*.conf`, `input_p63_val*.conf`, `input_p31_val*.conf`,
`input_p127_val*.conf`, `input_p255_val_gemm*.conf`, `input_p*_val*` といった
まとめ表記もある。これらは §2.1 と §2.2 の実体名の集合を指す。`conf_perf_p*` は
現行 `namelists/perf_p*`、`input_p*_val*` は `namelists/val_p*` に読み替える。

## 4. 現行の命名規則（要約）

正典は `namelists/README.md`。ここは索引としての要約に留める。

```
{purpose}_p{order}_{kernel}[_{qualifiers}].conf
```

- `purpose`: `demo`（極小スモーク）/ `perf`（公表メッシュの時間測定）/
  `val`（point-varying 係数の検証、`SCALE_DG_VARYING_COEFF=1`）/
  `acc`（時間積分後の精度、`SCALE_DG_ACCURACY=1`）
- `order`: 7, 15, 31, 63, 127, 255, 511, 575, 767, 1023
- `kernel`: `openacc_split`, `split`, `fused`, `fused_tc`, `fused_dfma`,
  `gemm`, `gemm_cute`, `gemm_fused`, `gemm_ozaki1`, `gemm_ozaki2`
- `qualifiers`: 既定でないときのみ `emu`（`CublasEmulation=.true.`）,
  `ne{N}`, `n{N}`

旧規則との対応の要点:

- `conf_perf_` → `perf_`、`input_..._val` → `val_`。
- `_tc` → `_fused_tc`、`_gf` → `_gemm_fused`、`_ref` → `_split`。
- 旧名の `kernel` 省略は現行では許されない。省略されていた旧名の経路は
  §2 の表で内容から確定させてある。
- `_ncu` コピーと一回限りの `Ne`/`nstep` 変種は `namelists/` に置かない
  （§3.1）。
