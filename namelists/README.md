# Namelists

Committed inputs are the samples needed to run the code and to reproduce
the published timing mesh. Experiment-only copies (`_ncu`, `_n400`, extra
`Ne`, load-audit) stay out of git: copy a `perf_` file and change `nstep`
or `DqdtKernel_Type`.

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
| `purpose` | `demo` (tiny smoke), `perf` (timing mesh), `val` (point-varying coefficient check) |
| `order` | polynomial order (`7`, `15`, …, `1023`) |
| `kernel` | `openacc_split`, `split`, `fused`, `fused_tc`, `gemm`, `gemm_cute`, `gemm_fused`, `gemm_ozaki1`, `gemm_ozaki2` |

Qualifiers only when they are not the default: `emu` (`CublasEmulation`), `ne{N}`, `n{N}`.

## Committed files

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
| `perf_p7_gemm_emu_n1.conf` | cuBLAS emulation timing (do not use `nstep=1000`) |
| `val_p7_*` / `val_p7_split.conf` / `val_p15_fused.conf` / `val_p15_split.conf` / `val_p15_gemm_fused.conf` / `val_p31_split.conf` / `val_p31_gemm_fused.conf` / `val_p63_split.conf` / `val_p63_fused_tc.conf` / `val_p127_fused.conf` / `val_p127_split.conf` / `val_p255_*` | numerical checks (`SCALE_DG_VARYING_COEFF=1`) |
| `val_p{511,575,767,1023}_gemm*.conf` | high-order GEMM / GEMM_FUSED / emu |

Another path on the same mesh: copy the `perf_` file and change `DqdtKernel_Type` only. Do not change `NeX/NeY/NeZ`, `PolyOrder`, `dt`, or `nstep` when comparing kernels.
