# 64-bit indexing boundary validation

> **Status:** `current`（初出 2026-08-28） — **現行ツリーに対する主張**として読んでよい文書。古くなったら更新する。

## 1. 結論

高次数で default integer の境界を越える可能性がある、次の2つのホスト側
extent / offset 計算を64-bit安全にした。

- `update_halo` の dummy argument を `f(Np*NeA)` から assumed-size `f(*)` にする。
- `CUDAFORTRAN_GEMM_FUSED` の `Escale` 方向別 pointer offset `Np*Ne` を、乗算前に
  `std::int64_t` へ昇格する。

変更前後を p=7、p=15、p=511 で比較した。全 owned `dqdt` は各次数で bitwise
identical、対象 device kernel の SASS と resource usage も同一だった。device
tendency の差は −0.19% から +0.02% の範囲で、有意な性能低下はない。このため
2変更を採用する。

## 2. 変更の境界

Fortran の dummy extent `Np*NeA` は、p=1023、Ne=1 で
`Np=1024^3=2^30`、`NeA=2` となり、default signed integer の範囲を越える。
これは caller が既に所有する配列を一次元表示するための宣言であり、halo loop が
実際に参照する最大添字は `Np*Ne+NhaloNode` である。assumed-size にすることで、
実行時の添字やデータ配置を変えずに不要な extent 積を除いた。

GEMM_FUSED の z epilogue は `Escale(:,:,1:3)` の先頭を
`escale + npoint` と `escale + 2*npoint` で求める。p=1023、Ne=1 では
`npoint=2^30` 自体は32-bitに収まるが、`2*npoint=2^31` が溢れる。変更後は

```c++
const std::int64_t npoint = std::int64_t{Np} * Ne;
```

として乗算と後続の pointer arithmetic を64-bitで評価する。device kernel 内の
要素・面・batch 添字は変更していない。したがって、これは p=1023 の実行可能性を
保証する変更ではなく、ホスト側で起動前に誤った pointer を形成する既知の境界を
取り除く変更である。メモリ容量、CUTLASS の grid/batch 上限、残る32-bit添字は
別途満たす必要がある。

## 3. 測定条件

- 日付: 2026-08-28（Asia/Tokyo）
- ブランチ: `feature/p511`
- baseline commit: `1107de3489654a8008598ee7aaa0ea9e0da66e72`
- candidate: 上記 commit に本レポートの2ソース変更を適用した working tree
- GPU: NVIDIA GB200、189471 MiB
- driver: 580.173.02
- compiler: NVIDIA HPC SDK 26.3
- target: CUDA Fortran `cc100`、C++ CUDA `sm_100`
- Slurm job / profiler: なし（login-node GPUで通常実行、`cuobjdump` のみ使用）
- CUDA graph: off
- timing: CUDA device event と end-to-end Main を別々に記録

ビルドは両版で同じコマンドを使い、比較中はそれぞれ `/tmp` に frozen copy を置いた。

```bash
module load nvhpc
make clean
make -j4 CUDA=1 GPUFLAGS=-gpu=cc100 GPUNVCCFLAGS=-arch=sm_100 \
  CUTLASS_HOME=/data1/rkp00015/rku00044/scale-dg-kernel-extraction/third_party/cutlass
```

**注記（2026-09-09）: 上の commit 表記は rebase 前のもので、現行履歴では解決できない。**
baseline の `1107de3489…` は commit オブジェクトとしては残っているが、どのブランチからも
到達不能である（`git branch -a --contains 1107de3` が空）。ブランチ `feature/p511` も
現存しない（現行で最も近い名前は `feature/p511-gemm-fused`）。また本レポートの 2 変更が
実際に入った commit が本文に書かれていない。それは `eb46c9c`
「Make high-order indexing boundaries 64-bit safe」（2026-08-28、
`cuda_cutlass_gemm_fused.cu` / `mod_mesh.f90` と本レポート・`reports/README.md`）である。
したがって現行履歴で candidate に当たるのは `eb46c9c`、baseline に当たるのはその親
`0980bdc` になる。**ただし `0980bdc` は `1107de3` と同じ subject「Add p=511 GEMM paths」
でありながら同一ツリーではない**: 両者の差は木全体で 30 ファイル・8261 挿入 / 444 削除
（`git diff --shortstat 1107de3 0980bdc`）、レポートや conf を除いたソース
（`*.cu` / `*.cuf` / `*.f90` / `*.h`）だけでも 15 ファイル・5134 挿入 / 424 削除であり、
`0980bdc` 側には Ozaki 実装（`cuda_ozaki1_gemm.cu` など）を含む後続 19 commit 分の内容が
畳み込まれている。`0980bdc` は「現行履歴での対応位置」であって §4 / §5 の測定を取った
ツリーそのものではないので、測定の厳密な再現には rebase 前の `1107de3` を直接
checkout する必要がある。

## 4. 性能結果

入力は同一体積 DOF の p=7 `Ne=32^3`、p=15 `Ne=16^3` と、p=511
`Ne=1` を使った。p=7 / p=15 は `CUDAFORTRAN_FUSED_TC`、p=511 は
`CUDAFORTRAN_GEMM_FUSED` である。p=7 / p=15 は `nstep=1000`,
`WarmupStep=20`（2940 measured RK stages）、p=511 は `nstep=50`,
`WarmupStep=5`（135 measured RK stages）。表は反復測定の中央値である。

**注記（2026-09-09）: §4 と §6 は namelist ファイル名を記録しておらず、§6 のとおり当時の
設定ファイルは削除済みなので、入力は本文のパラメータから再構成するしかない。**
現行 `namelists/` で格子と経路が一致するのは `perf_p7_fused_tc.conf`（`NeX=NeY=NeZ=32`、
`CUDAFORTRAN_FUSED_TC`）、`perf_p15_fused_tc.conf`（16^3、同経路）、
`perf_p511_gemm_fused.conf`（1^3、`CUDAFORTRAN_GEMM_FUSED`）の 3 本で、いずれも
`UseCudaGraph = .false.` は本文の条件と一致する。**ステップ数だけは一致しない**:
3 本とも `nstep = 20` で、`WarmupStep` は p=511 が 2、p=7 / p=15 は未指定（既定 1、
`main.f90:47`）である。本文の `nstep=1000` / `WarmupStep=20`（p=7 / p=15）と
`nstep=50` / `WarmupStep=5`（p=511）は上書きが要る。§6 で使った環境変数
`SCALE_DG_DUMP_DQDT` と `SCALE_DG_VARYING_COEFF` は現行の `main.f90`（それぞれ :590 と
:340 / :506）にも残っているので、検証手順そのものはそのまま再実行できる。

| order / path | baseline device | candidate device | 差 | baseline Main | candidate Main | 差 |
|---|---:|---:|---:|---:|---:|---:|
| p=7 `FUSED_TC` | 270.903 µs/stage | 270.557 µs/stage | −0.128% | 1074.520 µs/step | 1072.110 µs/step | −0.224% |
| p=15 `FUSED_TC` | 327.922 µs/stage | 327.283 µs/stage | −0.195% | 1249.820 µs/step | 1248.260 µs/step | −0.125% |
| p=511 `GEMM_FUSED` | 12.61844 ms/stage | 12.62089 ms/stage | +0.019% | 39.4813 ms/step | 39.4900 ms/step | +0.022% |

p=7 は baseline 5回 / candidate 3回、p=15 と p=511 は各3回測定した。
最大差は0.22%で符号も一定せず、起動・実行揺らぎの範囲である。

## 5. Device code と resource usage

`cuobjdump --dump-resource-usage` と `cuobjdump --dump-sass` で frozen executable
を比較した。

| kernel | baseline | candidate | SASS body SHA-256 |
|---|---|---|---|
| `update_halo` | REG 14 | REG 14 | `6f45e27b828afd2231ec2e15914a9208a2ad7a0bd044f2f8a9084c22a6577d7b` |
| p=15 `tendency_fused_p15_tc` | REG 64, SHARED 48896 B | REG 64, SHARED 48896 B | `6a81e8f56d53a435604ca26f6104aa75dbf225bced99677ee9dd830c557b0eed` |
| CUTLASS device kernels 全体 | 同一 | 同一 | `2760281774c6759c2333caa06f3a00fd7a632dc55018dcc3ac7f7c5beb58d2ec` |

各 SHA-256 は baseline と candidate で一致した。`f(*)` は device loop の
実添字を変えず、`std::int64_t` は kernel launch 前のホスト側 pointer 計算にしか
使われないことを、生成コードでも確認できる。

## 6. 数値検証

最初の RK stage の全 owned `dqdt` を `SCALE_DG_DUMP_DQDT` で出力し、変更前後を
`cmp` で比較した。定数係数だけへの特殊化を見逃さないよう、検証時は
`SCALE_DG_VARYING_COEFF=1` を指定した。

| order / path | owned 点数 | dump size / 版 | 結果 |
|---|---:|---:|---|
| p=7 `FUSED_TC` | 16,777,216 | 419,430,400 B | bitwise identical |
| p=15 `FUSED_TC` | 16,777,216 | 419,430,400 B | bitwise identical |
| p=511 `GEMM_FUSED` | 134,217,728 | 3,355,443,200 B | bitwise identical |

この比較は各次数で同じ path の baseline と candidate を比較するものなので、縮約順の
差はなく完全一致が期待値であり、結果もその通りだった。検証 dump、設定ファイル、
frozen executable は確認後に削除した。

**注記（2026-09-09）: 削除した設定ファイルの再構成については §4 の注記を参照。**

## 7. Build validation

変更後に非CUDA build と CUDA build をそれぞれ clean build し、stub と
CUDA Fortran / C++ interface がともに正常にビルドされることを確認した。最終的な
worktree にはタスクに対応する CUDA executable を残した。
