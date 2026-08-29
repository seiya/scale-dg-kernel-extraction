# Repository Guidance

## Purpose

This repository is a performance-oriented extraction of the SCALE-DG
three-dimensional DG advection kernels. Optimizations must preserve the
numerical operations and data semantics of the extracted implementation.
Performance improvements are not valid if they solve a more specialized
problem than the original array-based interface.

## What This Work Is For

The goal is **not only to find the fastest path**. It is to **model the
performance and publish it**. A result that identifies the fastest kernel but
cannot say *why* it is fastest is only half of the deliverable.

Three consequences that override the usual "is it worth it?" instinct:

- **Do what can be done.** Do not stop an investigation because the remaining
  speedup would be small, or because the winning path is already known and
  would not change. An optimization that loses is still a data point in the
  model, and a knob that turns out not to matter is a published constraint.
  "Not worth it for performance" is not a reason to stop; "there is nothing
  left that can be measured" is.
- **Close the unknowns.** An unexplained measurement is a defect in the model,
  not an acceptable loose end. If a bottleneck cannot be identified from the
  login node, that is a reason to submit the `ncu` / `nsys` job, not a reason
  to write "not diagnosable without a profiler" and stop. Every kernel that
  goes into a report should have a stated, measured reason for its speed --
  which hardware resource binds it, at what fraction of what roof, and what
  the ablation that proves it was.
- **Negative results are results.** Every rejected candidate belongs in the
  report with its number and its mechanism, so the same road is not walked
  twice and so the model has both sides of each trade-off. Ablations that
  isolate a cost (remove the mma, remove the epilogue, remove the global
  loads) are first-class evidence and should be recorded even when the
  variant is thrown away.

None of this relaxes the numerical contract below. A candidate that would
specialize the extracted array interface stays out of scope no matter how much
it would explain or how fast it would be; record it as out of scope and why.

## Numerical Contract

- Treat `q`, `u`, `v`, and `w` as pointwise fields. The benchmark currently
  initializes constant velocity values, but callers are allowed to provide a
  different value at every point.
- Treat `Escale` as a pointwise, per-element, per-direction field.
- Treat `normal_fn` and `Fscale` as face-point, per-element fields.
- Do not replace any of these arrays with a representative scalar such as
  `u(1,1)` or `Escale(1,1,:)`.
- The volume terms are derivatives of the pointwise fluxes:
  `D(q*u)`, `D(q*v)`, and `D(q*w)`. In general, they are not equivalent to
  `u*D(q)`, `v*D(q)`, and `w*D(q)`.
- Evaluate the numerical flux on all six faces using the M- and P-side values,
  `VMapM`, `VMapP`, `normal_fn`, and `Fscale`. Do not assume that three faces
  are identically zero based on the velocity used by a benchmark input.
- Fusing kernels, staging values in shared memory, specializing polynomial
  order, or changing launch geometry is allowed only while the above semantics
  remain intact.
- Keep halo data valid. Velocity halos are initialized before time stepping;
  the evolving `q` halo is updated at each required Runge--Kutta stage.

## Implementation Path Roles

The fastest production paths are `CUDAFORTRAN_GEMM_FUSED` and
`CUDAFORTRAN_FUSED_TC` (which one wins depends on polynomial order). The
other CUDA paths exist to isolate one axis at a time. Do not independently
speed-optimize a control path.

- `CUDAFORTRAN_GEMM`: unfused volume GEMM with cuBLAS. Surrounding kernels
  (volume flux, face flux, `separable_lift_assembly`, z written into `dqdt`)
  are the reference layout for library comparisons.
- `CUDAFORTRAN_GEMM_OZAKI1` / `CUDAFORTRAN_GEMM_OZAKI2`: the same unfused
  driver as `CUDAFORTRAN_GEMM`. Only the three volume GEMMs are replaced.
  Compare them to native cuBLAS and to `CublasEmulation` on that same
  driver. Do not put Ozaki on the CUTLASS fused paths.
- `CUDAFORTRAN_GEMM_CUTE`: same CUTLASS volume-GEMM tiles and
  `Nq<=64` cuBLAS-x switch as `CUDAFORTRAN_GEMM_FUSED`, but unweighted
  epilogues and a separate lift/assembly kernel. `GEMM` vs `GEMM_CUTE` is
  the library/mainloop difference. `GEMM_CUTE` vs `GEMM_FUSED` is the
  fusion package (at `Nq>64` that package includes Escale forwarding and
  folding `deriv_x` into y). Do not retune CUTE on its own.
- `CUDAFORTRAN_GEMM_FUSED`: the fused-epilogue production GEMM path.
  CUTLASS tile types live in `VolumeGemmSet` in `cuda_cutlass_gemm_fused.cu`.
- `CUDAFORTRAN_FUSED`: C++ fused kernels in `cuda_dg_kernels_tc.cu` with
  `UseTc=false` (DFMA on the MMA fragment layout). It is the Tensor Core
  ablation, not a separately optimized CUDA-core kernel. Geometry constants
  are in `fused_kernel_geom.h`.
- `CUDAFORTRAN_FUSED_TC`: the same source with `UseTc=true`
  (`mma.sync.m8n8k4`). Changing that file always changes both FUSED paths.

A change that would make FUSED and FUSED_TC diverge (except the inner
product) is a defect. A change that would make GEMM_CUTE's GEMM launches
diverge from GEMM_FUSED's mainloop tiles or `Nq<=64` x-library switch is
likewise a defect.

## Important Files

- `main.f90`: configuration, field initialization, time stepping, and GPU data
  lifetime.
- `mod_advect3d_eq.f90`: tendency dispatch and work-array management.
- `mod_cuda_dg_kernels.cuf`: CUDA Fortran wrappers, GEMM drivers, split kernels.
- `cuda_dg_kernels_tc.cu`: hand-written fused C++ kernels (`FUSED` / `FUSED_TC`).
- `fused_kernel_geom.h`: shared fused-kernel geometry constants.
- `mod_cuda_dg_kernels_stub.f90`: matching non-CUDA interfaces.
- `mod_mesh.f90`: mesh, mappings, halo handling, and p=255 operator generation.
- `mod_dg_optr_kernel_opt1.F90.erb`: source template for optimized DG kernels.
- `mod_dg_optr_kernel_opt1.f90`: generated file; update the template when the
  generated kernels must change, then regenerate with `make`.
- `reports/`: committed performance and optimization reports. See
  `reports/README.md` for the index and the current fastest path per
  polynomial order.

## Builds

Before an OpenACC or CUDA Fortran build, load the NVIDIA HPC SDK environment.
Use whichever of the following module names is available on the target system:

```bash
module load nvhpc
# or
module load nvhpc-hpcx
```

The same module load is required inside Slurm profiling jobs before running
NVIDIA profiling tools. Do not assume that a module loaded in the login shell
is inherited by a batch job.

Build modes use incompatible objects and module files, so run `make clean`
when switching modes.

```bash
# CPU/OpenMP
make clean
make

# OpenACC
make clean
make ACC=1

# CUDA Fortran
make clean
make CUDA=1
```

On RIKYU GB200 nodes, use an explicit target when building away from an
allocated GPU node:

```bash
make clean
make CUDA=1 GPUFLAGS=-gpu=cc100
```

Leave the working executable in the build mode relevant to the current task.

## Validation

- A constant-velocity benchmark is not sufficient to validate an optimized
  tendency kernel. Also compare against `OPENACC_ASIS`, `OPENACC_SPLIT`, or
  `CUDAFORTRAN_SPLIT` with deliberately point-varying `u`, `v`, `w`,
  `Escale`, `normal_fn`, and `Fscale`.
- Compare the complete owned `dqdt(:,1:Ne)` field, not only min/max values or
  elapsed time. Differences should be at floating-point roundoff level.
- For p=7, validate `CUDAFORTRAN_FUSED` against a split implementation.
- For p=255, test both the intended `Ne=1` case and an `Ne>1` smoke case when
  memory permits. The p=255 path currently requires `CUDAFORTRAN_FUSED`,
  `CUDAFORTRAN_FUSED_TC`, `CUDAFORTRAN_GEMM`, `CUDAFORTRAN_GEMM_FUSED`, or
  `CUDAFORTRAN_GEMM_CUTE`.
- After interface changes, verify both a CUDA build and a non-CUDA build so the
  stub remains synchronized.
- Run `git diff --check` before committing.

Ordinary GPU execution is available directly on the login node. Nsight Systems
(`nsys`) and Nsight Compute (`ncu`) must instead be run through a Slurm job
submitted with `sbatch`; do not launch either profiler directly on the login
node. Every `nsys` invocation needs `export DEBUGINFOD_URLS=` in the job and
`--resolve-symbols=false` on the command line; without both, symbol resolution
never finishes. Bound each profiler invocation with `timeout` so that a hang
costs one attempt instead of the whole allocation, and profile a frozen copy of
the executable rather than `scale-dg_extraction`, which a concurrent `make`
would relink under the running process.

Set that `timeout` from what the run actually takes, not generously: individual
GPU nodes go bad, and on a bad one the profiler produces no output at all
rather than running slowly, so a long bound only delays the retry. A profiling
job that normally finishes in under a minute has been seen to sit for forty
minutes on one node and return an empty report, while the same command on three
other nodes took 42-43 seconds (`p127_gap_study.md` §13.8). Give each profiler
invocation a few times its known duration, and when one times out, check
`sacct -j <id> --format=NodeList` first and resubmit with
`#SBATCH --exclude=<node>` before changing the profiler command.

`nsys` cannot profile the CUDA graph path (`UseCudaGraph = .true.`); it hangs
and the report contains no GPU trace.
Profile that configuration with `UseCudaGraph = .false.`, whose kernels and
order are identical.

Keep the input configuration identical when comparing implementations.
Do not change `NeX/NeY/NeZ`, `PolyOrder`, `dt`, or `nstep` as part of a
kernel-only performance comparison unless the experiment explicitly studies
that change.

## Profiling and Performance Reports

- Distinguish CUDA device-event time from end-to-end wall time.
- Record the exact input file, executable commit, GPU target, and profiler
  command for every comparison.
- Reprofile after changes to the numerical work or data traffic; results from a
  scalar-specialized kernel are not comparable to the array-correct kernel.
- For FLOP/s and bandwidth reports, show theoretical operation/byte counts and
  profiler-measured counts separately, including the assumed hardware peaks.
- On GB200 the FP64 Tensor Core peak equals the FP64 CUDA-core peak
  (40.1 TFLOP/s); the 2x advantage that Ampere and Hopper had is gone. Use the
  same denominator for CUDA-core and Tensor Core paths, and do not expect a
  Tensor Core rewrite to raise the arithmetic ceiling.
- `--set basic` does not collect Memory Workload Analysis, so it cannot show
  shared-memory bank conflicts. Add `--set full` or the
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_*` metrics when a kernel
  is L1/TEX bound.

### reports/

Finished reports live in `reports/` and are committed. Working notes,
profiler output, and Slurm logs are not; see the commit rules below.

- Read `reports/README.md` before starting a performance task. It states which
  path is currently fastest at each polynomial order, which is the thing most
  often out of date elsewhere.
- Every report states the commit, Slurm job, input file, and GPU it was
  measured on. Keep that habit for new reports.
- When a change invalidates a published conclusion, update the affected
  reports in the same commit as the code change, or in the commit that
  measures it. Do not rewrite the measured tables: leave them as recorded,
  state the commit they belong to, and add a note where the conclusion
  changed. Past numbers are evidence, not claims about the current tree.
- `reports/README.md` is the index. Add a row when adding a report.

## Working Tree and Commits

- Preserve unrelated user changes and untracked profiling artifacts.
- Do not commit Slurm job scripts, `slurm-*.out`, Nsight reports, profiler text
  or CSV output, or ad-hoc analysis Markdown unless the user explicitly asks.
  Reports under `reports/` are the exception: they are tracked, and updating
  them is part of the change that invalidates them.
- Stage source and documentation files explicitly rather than using
  `git add -A`.
- Before amending or rebasing, identify the commit that first introduced the
  behavior and keep unrelated history unchanged.
