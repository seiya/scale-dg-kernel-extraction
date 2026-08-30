# 具体的なコマンドと落とし穴

すべて repo ルート (`/data1/rkp00015/rku00044/scale-dg-kernel-extraction`) から。

## ビルド

```bash
module load nvhpc-hpcx      # または nvhpc
make clean && make CUDA=1 GPUFLAGS=-gpu=cc100    # GB200 (cc100)、GPU ノード外でも通る
```

- モード切替（CPU / `ACC=1` / `CUDA=1`）は必ず `make clean` を挟む。
  オブジェクトと module ファイルに互換性が無い。
- インタフェースを変えたら **CUDA ビルドと非 CUDA ビルドの両方**を通し、
  `mod_cuda_dg_kernels_stub.f90` を同期させる。
- `mod_dg_optr_kernel_opt1.f90` は生成物。`mod_dg_optr_kernel_opt1.F90.erb`
  を直して `make`。
- 作業終了時は、そのタスクに関係するビルドモードの実行ファイルを残す。

## 計測（login node で直接実行可）

```bash
for i in 1 2 3; do ./scale-dg_extraction conf_perf_p63_tc.conf | tail -8; done
```

- 読む値: `Main per step:`（end-to-end）と `Cal_tend:`（device）。両者を混同しない。
- `UseCudaGraph = .true.` では `Cal_tend:` は "not measured (graph)" になる。
- `WarmupStep`（既定 1、graph 経路では 2 以上に自動昇格）で先頭ステップは計時外。
  `nstep` は変えない。
- 3〜5 回の中央値。レンジが重なる差は「差が無い」と書く。
- **採否を決める計測は sbatch で占有した GPU 上で**、変種を交互に 10〜12 回。
  ログインノードは他人と共有で、同じ実行ファイルでも 0.5% ほどぶれる。
  1 % 未満の差はこれをやらないと判定できない。
- sbatch のあと `squeue -j <id>` でキューに残っているか見る。残っていなければ
  `sacct -j <id>` で完了を確認して出力を読む。
- **conf を変えて比較しない**。`NeX/NeY/NeZ`, `PolyOrder`, `dt`, `nstep` 固定。

## 数値検証

```bash
SCALE_DG_VARYING_COEFF=1 SCALE_DG_DUMP_DQDT=/tmp/.../ref.txt \
  ./scale-dg_extraction input_p63_val_ref.conf
SCALE_DG_VARYING_COEFF=1 SCALE_DG_DUMP_DQDT=/tmp/.../new.txt \
  ./scale-dg_extraction input_p63_val_tc.conf
# ビット一致は不要。cmp の不一致だけで落とさない。max abs / 相対差を見る。
paste ref.txt new.txt | awk '{d=$1-$2; a=d<0?-d:d; if(a>m){m=a; r=$1}} END{print m, m/(r<0?-r:r)}'
```

- **ビット一致は要求しない。** 合格は `dqdt` 全点の差が丸め誤差レベルであること
  （典型は max abs 1e-12〜1e-15）。加算順や FMA が変わってもよい。
- `SCALE_DG_VARYING_COEFF=1` は `u,v,w,Escale,normal_fn,Fscale` を点ごとに
  変動させる。**定速度ベンチだけでは検証にならない**（`AGENTS.md`）。
- 参照は `OPENACC_ASIS` / `OPENACC_SPLIT` / `CUDAFORTRAN_SPLIT`。
- 比較対象は `dqdt(:,1:Ne)` 全点。min/max や実行時間では不十分。
- p=255 は `Ne=1` と、メモリが許せば `Ne>1` のスモークも。
- `SCALE_DG_DUMP_Q` は q の同様のダンプ。

## ncu / nsys（**必ず sbatch。login node で直接起動しない**）

```bash
#!/bin/bash
#SBATCH --job-name=ncu_p63tc
#SBATCH --gpus=1
#SBATCH --time=02:00:00
module load nvhpc-hpcx
export DEBUGINFOD_URLS=          # nsys に必須。ncu でも害は無い
EXE=./scale-dg_extraction.p63tc  # 凍結コピー。並行 make に relink されないように
OUTDIR=./output; mkdir -p $OUTDIR

timeout 3600 ncu --set full --csv --kernel-name-base function --rename-kernels 0 \
  -s 12 -c 6 $EXE conf_perf_p63_tc_ncu.conf \
  > $OUTDIR/ncu_p63_tc.csv 2> $OUTDIR/ncu_p63_tc.err
```

- 実行ファイルは `cp scale-dg_extraction scale-dg_extraction.<tag>` で凍結してから
  プロファイルする。
- 各プロファイラ呼び出しを `timeout` で囲む（ハングで割り当てを失わない）。
- `-s`/`-c` で定常状態の数ローンチだけを採る。ncu 用 conf は `nstep` が短い。
- `--set basic` は Memory Workload Analysis を採らない → **バンクコンフリクトが
  見えない**。`--set full` か明示メトリクス。
- **`ncu` の時間で採否を決めない**（`SKILL.md` 手順 3）。クロック固定で global の
  レイテンシがサイクル数で短く見えるため、shared・命令を削る変更は過大評価、
  global の待ちを隠す変更は過小評価になる。**同一ジョブ内に変種を並べて**機構を
  同定し、採否は占有 GPU 上の実時間 A/B で決める。
- DRAM の実バイトを見たいときは `dram__bytes_read.sum` / `dram__bytes_write.sum` と
  `lts__t_sector_hit_rate.pct` を採る。**アブレーションが「アクセスの飛び方」だけ
  でなく「触るデータ量」まで変えていないか**を、これで必ず確かめる
  （`p63_gap_study.md` §20.6 は、これを確かめずに天井を 2 倍見積もった例）。
- `nsys` には `--resolve-symbols=false` が必須（`DEBUGINFOD_URLS=` と両方）。
  **`nsys` は CUDA graph 経路 (`UseCudaGraph=.true.`) を採れない**（ハングして
  GPU trace が空）。`UseCudaGraph=.false.` で採る（カーネルと順序は同一）。
- `gen_ncu_cmds.sh` は nsys の `--stats` ログから、カーネルごとの
  eval 可能な ncu コマンドを生成する（CUTLASS の長い symbol 対策込み）。

### よく使うメトリクス

```
# stall 内訳（この repo の job スクリプトで実際に使っている形）
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
# shared / global の L1/TEX 分離とバンクコンフリクト
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,
l1tex__data_pipe_lsu_wavefronts_mem_global.sum,
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,
l1tex__throughput.avg.pct_of_peak_sustained_elapsed
# global の効率
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
# 命令数・時間
smsp__inst_executed.sum, gpu__time_duration.sum
```

## SASS

```bash
cuobjdump -sass scale-dg_extraction | sed -n '/<kernel name>/,/^$/p' | less
nvdisasm -c ...     # 制御フロー付き
```

見るべき例: `LDS.64` / `STS.128` のマージ（隣接 8 B ストア 2 本は `STS.128` に
まとまるので、bit 0 を使う swizzle はストア側からは見えない）、`DMMA.8x8x4`
への展開、レジスタ数とスピル (`LOCAL` へのアクセス)。

## ハードウェアの前提（GB200）

- FP64 CUDA core ピーク = FP64 Tensor Core ピーク = **40.1 TFLOP/s**。
  Ampere/Hopper にあった TC の 2 倍優位は無い。**同じ分母**で比較し、
  TC 化で演算天井が上がるとは考えない。
- shared 8 B アクセスのコンフリクトフリー条件は「32 レーンで `d mod 32` が相異」
  ではなく **「半ワープ 16 レーンで `d mod 16` が相異」**（`p63_gap_study.md` §16.4）。
- H100 (TSUBAME 4) では FP64 TC ピークが 2 倍あり、`CutlassMmaShape = "16x8x4"`
  を選ぶ。GB200 とは結論が違う（`h100_report.md`）。

## FLOP/s・帯域の報告

理論の演算数/バイト数と、プロファイラ実測値を**分けて**書く。仮定した
ハードウェアピークを明示する。
