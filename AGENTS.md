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

Across the polynomial orders measured so far the fastest path is either
`CUDAFORTRAN_GEMM_FUSED` or `CUDAFORTRAN_FUSED_TC`; see `reports/README.md`
for the current table. That is a measurement, not a role assignment.
`CUDAFORTRAN_FUSED` is a production path as well and is optimized as such.
The remaining CUDA paths are controls that exist to isolate one axis at a
time; do not independently speed-optimize a control path.

Numerical correctness and path identity are two independent requirements.
A candidate is eligible for production only if it satisfies both:

1. It preserves the Numerical Contract above.
2. It still implements the path selected by `DqdtKernel_Type` as defined in
   this section.

Full-field agreement, including bit-exact agreement, proves only the first
requirement. It does not make a library substitution, a split epilogue, or a
different algorithm valid for a named path. Path names are part of the
benchmark contract, not descriptive labels that may be reassigned to whichever
implementation is fastest.

- `CUDAFORTRAN_GEMM`: unfused volume GEMM with cuBLAS. Surrounding kernels
  (volume flux, face flux, `separable_lift_assembly`, z written into `dqdt`)
  are shared with `GEMM_CUTE`, the Ozaki paths and `GEMM_FUSED`, which is what
  makes those three comparisons one-axis ones (mainloop, GEMM internals,
  epilogue fusion). Shared does not mean unoptimized: the GEMM family is the
  paper's library baseline and a weak baseline overstates the fused paths'
  win, so optimize these kernels -- but land the change on all of them at
  once. The one fusion `GEMM` must not have is the epilogue itself, which is
  the axis `GEMM_FUSED` measures and which cuBLAS cannot express anyway.
  This sharing is structural, not a convention: `GEMM`, `OZAKI1` and `OZAKI2`
  are thin wrappers over one `cuda_cal_dqdt_gemm_unfused(backend, ...)` driver,
  and `GEMM_CUTE` / `GEMM_FUSED` over one
  `cuda_cal_dqdt_gemm_cutlass(fuse_epilogue, ...)` driver. Keep it that way:
  optimize a surrounding kernel through its shared launch helper (e.g.
  `launch_volume_flux`) so every GEMM path gets it at once.
- `CUDAFORTRAN_GEMM_OZAKI1` / `CUDAFORTRAN_GEMM_OZAKI2`: the same unfused
  driver as `CUDAFORTRAN_GEMM`. Only the three volume GEMMs are replaced.
  Compare them to native cuBLAS and to `CublasEmulation` on that same
  driver. Do not put Ozaki on the CUTLASS fused paths.
- `CUDAFORTRAN_GEMM_CUTE`: the unfused control for the CUTLASS GEMM path.
  It must use the same volume-GEMM mainloops, tile shapes, MMA shapes, stage
  counts, swizzles, order specializations, batch chunking, and per-GEMM
  library assignment as `CUDAFORTRAN_GEMM_FUSED`. Its epilogues are
  unweighted, z is materialized in `dqdt`, and a separate
  `separable_lift_assembly` kernel performs the final weighting and surface
  lift. `GEMM` vs `GEMM_CUTE` measures the library/mainloop difference. Do
  not retune CUTE on its own.
- `CUDAFORTRAN_GEMM_FUSED`: the fused-epilogue production CUTLASS GEMM path.
  It has the same volume-GEMM mainloops and the same per-GEMM library
  assignment as `GEMM_CUTE`. The last of its three volume GEMMs must fuse the
  final volume weighting and surface lift/assembly into its epilogue; it must
  not materialize all three derivatives and then call
  `separable_lift_assembly`. Which GEMM carries that epilogue is a
  measurement, not part of the
  definition: the epilogue reads the other two derivatives elementwise at the
  same node index, so x, y or z can each carry it by running last, and each
  offers a different face-point pair for tile-level reuse and a different
  register budget. z is the current carrier and the only one measured. The
  carrier cannot be cuBLAS, which has no epilogue. That is the only fixed
  point of the library assignment: which library runs each of the other two
  volume GEMMs is a measurement, and the assignment must be identical in
  `GEMM_CUTE` and `GEMM_FUSED`. There is no threshold in this document to
  honour: the assignment in the tree is whatever last measured fastest, and
  the gap studies record it. Beyond that required fusion, forwarding Escale
  factors into the other two volume GEMMs and folding one derivative into
  another belong to the fusion package as well: they are consequences of
  fusing, not mainloop differences, so `GEMM_FUSED`
  may carry them where `GEMM_CUTE` cannot, and which of them to carry at a
  given order is a measurement, not a rule stated here. With z as the carrier
  they are gated by `xy_weighted` (Escale into x/y, `deriv_x` into y); the
  orders where that measured faster are in the gap studies.
  CUTLASS tile types live in `VolumeGemmSet` (or an explicitly shared
  order-specialized set) in `cuda_cutlass_gemm_fused.cu`.
- `CUDAFORTRAN_FUSED`: CUDA-core fused kernels in `cuda_dg_kernels_fused.cu`
  (natural-order shared panels, length-`Nq` inner products). This is the
  paper's "CC fused" column, and it is a full optimization target: make it as
  fast as CUDA cores allow within that implementation style. What it must not
  do is adopt the TC path's implementation: that variant already exists and is
  measured as `FUSED_DFMA`, and merging the two would cost the iso-schedule
  ablation -- the only thing that separates the mma instruction from the
  schedule built around it. The paper's CC column is the fastest CC path, not
  a property of this one: `FUSED` currently holds it at every order measured,
  and if the TC schedule ever won, the answer is to report `FUSED_DFMA` as the
  fastest CC, not to rewrite `FUSED`. Selecting `FUSED` must launch these
  kernels. Never dispatch `FUSED_DFMA` or `FUSED_TC` in its place.
- `CUDAFORTRAN_FUSED_TC`: Tensor Core fused kernels in
  `cuda_dg_kernels_tc.cu` with `UseTc=true` (`mma.sync.m8n8k4`).
- `CUDAFORTRAN_FUSED_DFMA`: the same source as `FUSED_TC` with
  `UseTc=false` (DFMA on the MMA fragment layout). Iso-schedule ablation
  of the MMA instruction only. Selecting `FUSED_DFMA` must launch that
  instantiation; never treat it as `FUSED`.

A change that would make `FUSED_DFMA` and `FUSED_TC` diverge (except the
inner product) is a defect. Do not copy the TC fragment layout or z shared
roundtrip into `FUSED`.

For `GEMM_CUTE` / `GEMM_FUSED`, the following are defects even when they are
numerically correct and faster:

- giving the epilogue carrier to cuBLAS, which has no epilogue;
- implementing `GEMM_FUSED` as a cuBLAS/CUTLASS volume GEMM followed by a
  separate lift/assembly kernel;
- giving `GEMM_CUTE` and `GEMM_FUSED` different volume mainloop tiles,
  stage counts, swizzles, batch partitioning, order-specialized launch
  geometry, or per-GEMM library assignment.

Polynomial-order specialization is allowed, but it may specialize only the
shared CUTLASS mainloop/launch configuration and the path-appropriate
epilogue. For example, an `Nq==32` y tile is valid only when both `GEMM_CUTE`
and `GEMM_FUSED` use that tile; an `Nq==32` cuBLAS-z plus separate-lift branch
is not a valid `GEMM_FUSED` implementation.

Out-of-role variants may be built as temporary ablations to measure a ceiling.
Label them explicitly as out of scope, record their performance and mechanism,
and remove them from production dispatch. Do not publish their timing as the
named path's production result or use it to update the fastest-path table.
Changing a path definition requires an explicit edit to this section; a local
performance win must never be treated as implicit permission to redefine it.

## Important Files

- `main.f90`: configuration, field initialization, time stepping, and GPU data
  lifetime.
- `mod_advect3d_eq.f90`: tendency dispatch and work-array management.
- `mod_cuda_dg_kernels.cuf`: CUDA Fortran wrappers, GEMM drivers, split kernels.
- `cuda_dg_kernels_tc.cu`: iso-schedule fused C++ kernels (`FUSED_DFMA` / `FUSED_TC`).
- `cuda_dg_kernels_fused.cu`, `cuda_dg_kernels_fused_highp.cu`: CUDA-core fused kernels (`FUSED`).
- `fused_kernel_geom.h`: shared fused-kernel geometry constants.
- `mod_cuda_dg_kernels_stub.f90`: matching non-CUDA interfaces.
- `mod_mesh.f90`: mesh, mappings, halo handling, and p=255 operator generation.
- `mod_dg_optr_kernel_opt1.F90.erb`: source template for optimized DG kernels.
- `mod_dg_optr_kernel_opt1.f90`: generated file; update the template when the
  generated kernels must change, then regenerate with `make`.
- `namelists/`: committed sample and published-mesh Fortran namelists. Names are
  `{purpose}_p{order}_{kernel}[_{qualifiers}].conf`; see `namelists/README.md`.
  Do not add `_ncu` copies or one-off `Ne` / `nstep` variants.
- `jobs/` is local (gitignored). Durable profiler commands go in `reports/`.
- `reports/`: committed performance and optimization reports. See
  `reports/README.md` for the index and the current fastest path per
  polynomial order.
- `.claude/skills/dg-optimize/` and `.cursor/skills/dg-optimize/`: the
  kernel-optimization procedure for Claude Code and Cursor. Same tree
  (Cursor path is a symlink). It follows this file and does not add rules.

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
  Bit-exact identity is not required: a change in FMA contraction or
  reduction order is acceptable if the full-field difference stays at
  roundoff. Do not reject a candidate solely because `cmp` / `filecmp`
  fails.
- For p=7, validate `CUDAFORTRAN_FUSED` against a split implementation.
- For p=255, test both the intended `Ne=1` case and an `Ne>1` smoke case when
  memory permits. The p=255 path currently requires `CUDAFORTRAN_FUSED`,
  `CUDAFORTRAN_FUSED_TC`, `CUDAFORTRAN_FUSED_DFMA`, `CUDAFORTRAN_GEMM`,
  `CUDAFORTRAN_GEMM_FUSED`, or `CUDAFORTRAN_GEMM_CUTE`.
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
- **`ncu` locks the clocks, and that biases every comparison it makes.** With
  the SM clock held down, a global memory access costs fewer SM cycles than it
  does at boost, so `ncu` prices shared-memory, register and instruction work
  relatively high and global-memory waits relatively cheap. It therefore
  overstates the gain from removing shared traffic or instructions, and
  understates the gain from spending them to hide a global load. Its kernel
  times also understate how much of a stage a DRAM-latency-bound kernel really
  costs.
  **Use `ncu` to identify the mechanism -- which stall moved, whether the
  conflicts went away, how many bytes DRAM actually moved -- and decide whether
  to adopt a change on wall time, from an interleaved A/B on a GPU the job
  owns.** Comparisons across two `ncu` jobs are not reliable either; put the
  variants in one job.
  Six measurements support this, five of them in one direction and one in the
  other, which is what pins the mechanism rather than a constant offset
  (`p63_gap_study.md` §19.10, §20.3, §20.6): a shared-conflict fix that `ncu`
  scored at −8.3% lost 1.0% of wall time; a prefetch that `ncu` scored at
  +8.3% won 0.51%; and the face flux kernel that `ncu` put at 10% of a stage
  is 13.45% of it.

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
- Do not commit Slurm job scripts (`jobs/` is gitignored), `slurm-*.out`, Nsight reports, profiler text
  or CSV output, or ad-hoc analysis Markdown unless the user explicitly asks.
  Reports under `reports/` are the exception: they are tracked, and updating
  them is part of the change that invalidates them.
- Stage source and documentation files explicitly rather than using
  `git add -A`.
- Before amending or rebasing, identify the commit that first introduced the
  behavior and keep unrelated history unchanged.
