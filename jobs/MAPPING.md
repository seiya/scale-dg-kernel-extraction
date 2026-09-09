# 旧ジョブスクリプト名 → 現行名の対応表

## 0. このファイルの役目と、`jobs/` の追跡方針

`reports/` 配下のレポートは、測定当時に使った Slurm / SGE / TCS ジョブスクリプトを
そのファイル名で引用する。`reports/README.md`:11 のとおり **本文の旧名は
書き換えない**（測定済みの記録は証拠であって、現在のツリーに対する主張ではない）。
その代わりに旧名から現行名を引くための受け皿がこのファイルである。
namelist 側は `namelists/MAPPING.md`。

### 大半が「現行名は分かるが git には無い」になる理由

`jobs/` は原則ローカルの作業場で、**`.gitignore` の allowlist（`/jobs/*` を無視し
`!/jobs/...` で個別に戻す）に列挙されたファイルだけが追跡される**。allowlist の
理由は AGENTS.md と `jobs/README.md` が書くとおり、**公表した機械間比は、その全
カラムを出したスクリプトが追跡されていなければ再現できない**から。それ以外——
一回限りのアブレーション、掃引、`ncu` / `nsys` プローブ——は追跡しない。

commit `490e94e`（branch `feature/cuda`）時点:

| | 数 |
|---|---|
| `ls -A jobs/` のエントリ | 339 |
| 追跡（`git ls-files jobs/`、`README.md` 含む） | 40 |
| 未追跡 | 299 |

（この表はこのファイルを allowlist に加える前の数である。`!/jobs/MAPPING.md`
を加えたので、現在の追跡は 41、未追跡は 298。）

したがって下の対応表は、**現行名が分かってもその実体は追跡されていない**という
行がほとんどになる。表には「追跡」列を置いてその区別を明示する。
（このファイル自身も `/jobs/*` に当たるので、追跡するには `.gitignore` に
`!/jobs/MAPPING.md` の行が要る。）

### 確認手順

- 現存: `ls jobs/`。追跡: `git ls-files jobs/`。無視判定: `git check-ignore -v`。
- 旧名の履歴: `git log --all --diff-filter=A -- 'jobs/<name>' '<name>'`。
- §1 の旧名 19 本は、**HEAD の祖先ではない** Cursor 作業ブランチの先頭コミット
  `6466dc0`（2026-08-29, "Cursor: Apply local changes for cloud agent"）に、
  リポジトリ**直下**の `job_*.sh` として実体がある。mainline では一度も追跡されて
  いない（`jobs/` は当時から ignore 対象で、直下の `job_*.sh` も未コミットだった）。
- 対応付けは推測ではなく**本文の突き合わせ**で行った。
  `git show 6466dc0:job_xxx.sh` と現行ファイルの `diff` に出るのは
  (a) `#SBATCH --job-name` の改名、(b) チェックアウトを探して `cd` する 2 行の追加、
  (c) namelist パスの `conf_perf_*.conf` → `namelists/*.conf` 書き換え、
  (d) 参照先の `./gen_ncu_cmds.sh` → `./jobs/gen_ncu_cmds.sh`
  だけで、測定内容（`ncu` / `nsys` のコマンド、カーネル指定、skip/count、
  経路のループ）は一致する。**内容で裏が取れないものは §2 に置き、対応を推測で
  埋めていない。**

## 1. 対応表（確定）

「追跡」列: **tracked** = allowlist にある / **untracked** = 現ツリーに実在するが
追跡外。19 本すべて現ツリーに実在する。

| 旧名（レポート本文の表記） | 現行名 | 追跡 | 引用レポート |
|---|---|---|---|
| `job_all.sh` | `jobs/sweep_p7_p255_all_paths.sh` | untracked | `overall_summary_report.md`, `tc_paper_survey_2407.09621.md` |
| `job_ncu_p15.sh` | `jobs/ncu_p7_p15_fused.sh` | untracked | `p15_gap_study.md` |
| `job_ncu_p15tc.sh` | `jobs/ncu_p15_fused_tc.sh` | untracked | `p15_gap_study.md` |
| `job_ncu_p15face.sh` | `jobs/ncu_p15_face.sh` | untracked | `p15_gap_study.md` |
| `job_ncu_occ.sh` | `jobs/ncu_p15_occupancy.sh` | untracked | `p15_gap_study.md` |
| `job_ncu_p31.sh` | `jobs/ncu_p31_fused_vs_gemm_fused.sh` | untracked | `p31_gap_study.md` |
| `job_ncu_p31b.sh` | `jobs/ncu_p31_gemm.sh` | untracked | `p31_gap_study.md` |
| `job_ncu_p63.sh` | `jobs/ncu_p63_gemm_fused.sh` | untracked | `p63_gap_study.md` |
| `job_ncu_p127.sh` | `jobs/ncu_p63_p127_p255_gemm_fused.sh` | untracked | `p127_gap_study.md` |
| `job_ncu_bankconflict.sh` | `jobs/ncu_bankconflict.sh` | untracked | `tc_paper_survey_2407.09621.md` |
| `job_ncu_tma_a0.sh` | `jobs/ncu_tma_a0.sh` | untracked | `tma_survey.md` |
| `job_ncu_tma_a0b.sh` | `jobs/ncu_tma_a0b.sh` | untracked | `tma_survey.md` |
| `job_ncu_tma_b0.sh` | `jobs/ncu_tma_b0.sh` | untracked | `tma_survey.md` |
| `job_ncu_tma_b0b.sh` | `jobs/ncu_tma_b0b.sh` | untracked | `tma_survey.md` |
| `job_nsys_p31.sh` | `jobs/nsys_p31_gemm.sh` | untracked | `p31_gap_study.md` |
| `job_nsys_p63.sh` | `jobs/nsys_p63_p255_gemm.sh` | untracked | `p63_gap_study.md` |
| `job_nsys_p127.sh` | `jobs/nsys_p63_p127_p255_gemm.sh` | untracked | `p127_gap_study.md` |
| `job_tsubame_ch7.sh` | `jobs/tsubame_ch7.sh` | untracked | `h100_report.md` |
| `job_tsubame_mma_shape.sh` | `jobs/tsubame_mma_shape.sh` | untracked | `sm90_mma_shape_survey.md`, `h100_report.md` |

注意: 改名後に対象が増えたものがある（`job_ncu_p15.sh` → `ncu_p7_p15_fused.sh`、
`job_ncu_p127.sh` → `ncu_p63_p127_p255_gemm_fused.sh`、`job_nsys_p63.sh` →
`nsys_p63_p255_gemm.sh`。同じ実行ファイルで別次数も採ったため）。旧名が指すのは
現行ファイルの**一部**であり、現行ファイルを走らせると当時より多くの構成を測る。

### 命名規則の変化

| | 旧 | 現行 |
|---|---|---|
| プロファイラ / 掃引 | `job_{ncu,nsys}_{topic}.sh` | `{tool}_{scope}_{topic}.sh`（`ncu_*`, `nsys_*`, `ab_*`, `time_*`, `sweep_*`） |
| 機械別ジョブ | `job_{machine}_{purpose}.sh` | `{machine}_{purpose}.sh`（`rikyu_*`, `tsubame_*`, `wisteria_*`） |
| 置き場所と起動 | リポジトリ直下、直下から投入 | `jobs/` 配下、スクリプト自身がチェックアウトを見つけて `cd` |
| namelist の参照 | `conf_perf_*.conf`（直下） | `namelists/*.conf` |

この 19 本が足した `cd` は `ROOT="$(cd "$(dirname "$0")/.." && pwd)"` の 2 行、
つまり **`$0` から引いている**。`$0` を使わずスケジューラの投入ディレクトリから
引くのは `tsubame_paths.sh` 系の機械間 harness だけで（SGE / TCS は `$0` が
スプールのコピーになる）、この 19 本は Slurm 専用なのでその区別は無い。

`job_` 接頭辞は現行規約には無い。`jobs/item1b/job_ab1.sh` などは旧規約の残骸では
なく、その作業ディレクトリ内のローカルな命名である。

## 2. 現存しないもの / 対応が確定できないもの

| 旧名 | 引用元 | 状態 |
|---|---|---|
| `jobs/job_p7_shwf.sh` | `tc_paper_survey_2407.09621.md`（2464 行・2706 行。引用箇所自身が「`jobs/` は gitignore なので ncu コマンドは本文に転記した」と書いている） | **現ツリーに無く、git 履歴にも無い**（`6466dc0` スナップショットにも入っていない）。未追跡のまま失われた。**現行名は不明**。同名ファイルが `.claude/worktrees/agent-accbc8a32a8a59f16/jobs/` にあるが、これはエージェント用 worktree のコピーで、本ツリーの対応物ではない。復元が要るならレポート本文に転記された `ncu` コマンドから |
| `job_*.sh`（総称、`reports/README.md`） | `README.md` | 旧規約の総称。実体は §1 の 19 本 |
| `jobs/*_ozaki.sh`（総称） | `ozaki2_implementation_report.md` のみ（561/566/572 行）。`cublas_emulation_survey.md` と `gemm_assignment_and_carrier.md` が持つのは総称ではなく実名 `jobs/acc_highorder_emu_ozaki.sh` で、部分一致の誤検出だった | `tsubame_ozaki.sh` / `wisteria_ozaki.sh` は **tracked で実在**。`rikyu_ozaki.sh` は存在せず、履歴にも無い。GB200 側の Ozaki 再測定は未追跡の `jobs/rikyu_recheck_ozaki.sh` が担当していると当該スクリプトのヘッダは述べるが、総称が GB200 の何を指していたかは**確定できない** |
| `st.sh` | `tc_paper_survey_2407.09621.md` | ジョブスクリプトではない。本文中の PTX 命令 `st.shared.f64` を機械抽出したときの**誤検出** |
| `output/s21/*.py`（総称） | — | `jobs/` ではなく出力ツリー側。`jobs/` の対応物は無い |

### 監査の指摘のうち、事実と異なっていたもの

- `{rikyu,wisteria}_cublas_emu.sh` と `rikyu_cublas_emu_nsys.sh` は「不在」ではない。
  `jobs/cublas_emulation_ab.sh` を共有ボディとする `rikyu_` / `tsubame_` /
  `wisteria_` の 3 ラッパも、その nsys 版 `rikyu_` / `tsubame_` も、
  **allowlist にあり追跡され、現ツリーに実在する**。
- `jobs/job_p7_shwf.sh` は「`job_` 接頭辞のまま `jobs/` に残る唯一の例外」ではない。
  上表のとおり現ツリーには無い。

## 3. 公表した機械間比を支えているのに未追跡のスクリプト（追跡候補）

allowlist の趣旨は「公表した機械間比は全カラムのスクリプトが追跡されていないと
再現できない」である。以下は**その趣旨に該当するのに allowlist に無い**もの。
すべて現ツリーに実在し（`ls jobs/`）、引用は `grep -rlF` で確認した。役割は各
スクリプト冒頭のコメントによる。`jobs/README.md` の追跡一覧には（追跡外なので）
名前が無い。

| ファイル（すべて untracked） | 引用レポート | 役割（スクリプト冒頭のコメントより） |
|---|---|---|
| `jobs/tsubame_p511_fusion_roof.sh` | `measured_dram_and_roofs.md`, `h100_report.md` | H100 の p=511、既定命令形での到達率比 `GEMM_FUSED / FUSED_TC` を 1 ジョブ・1 ノードで測る |
| `jobs/wisteria_a6_gemm.sh` | `a100_report.md` | A100 の `GEMM / FUSED_TC` デバイス時間比（`paper_notes/open_measurements.md` 1.2） |
| `jobs/gemm_library_ncu.sh` ＋ `jobs/rikyu_gemm_library_ncu.sh` | `h100_report.md` | ライブラリ / mainloop 軸（`GEMM_CUTE` 対 `GEMM`）。H100 用ボディと「同じスクリプトの GB200 側半分」の対で 1 つの機械間比を成す |
| `jobs/mma_axis_ncu.sh` ＋ `jobs/rikyu_mma_axis_ncu.sh` | `h100_report.md` | mma 命令軸（`FUSED_TC` 対 `FUSED_DFMA`）。同じく H100 ＋ GB200 の対 |
| `jobs/h100_p511_gemm_subtraction.sh` ＋ `jobs/rikyu_p511_gemm_subtraction.sh` | `h100_report.md` | p=511 の `ncu` カーネル合計とキャンペーン device 時間の差の内訳。GB200 側はクロック固定率が違う（0.53×）ことを使った対照 |
| `jobs/h100_16x8x8.sh` ＋ `jobs/rikyu_h100_16x8x8.sh` | `h100_report.md` | `CutlassMmaShape="16x8x8"` の H100 再評価と、採用条件である GB200 側の値付け |
| `jobs/tsubame_mma_shape.sh` | `h100_report.md`, `sm90_mma_shape_survey.md` | H100 の mma 形状スイープ |

`jobs/tsubame_ch7.sh`（`h100_report.md` §3 の測定）も同種だが、**すでに
`jobs/README.md` の "Not tracked" 節が「未追跡だが追跡ファイルから名前で参照されて
いる 3 つ」の 1 つとして扱っており**、`tsubame_paths.sh` がその後継である。ただしその節が書く復元元 commit
`9eed9e5` は誤りで、`9eed9e5` は `jobs/` を一切含まない（そこは
`h100_report.md` §3 が**測定した**ツリーの commit）。実体があるのは §1 と同じ
`6466dc0` のリポジトリ直下 `job_tsubame_ch7.sh` で、現行ファイルとの差は
チェックアウトを探す 24 行の追加だけである。ここでは重複して候補に挙げない。

追跡するか否かは方針判断であり、このファイルは**該当することの記録**にとどめる。
追跡する場合は `.gitignore` の `!/jobs/...` 行の追加と `jobs/README.md` の
追跡一覧への追記が同時に必要になる。
