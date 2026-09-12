#!/usr/bin/env python3
"""Leave-one-GPU-out transfer prediction for the SM sub-pipe model.

reports/subpipe_transfer_model.md defines the model.  This script is the
executable form of it, so that the hold-out claims in that report can be
re-run by a third party from the committed ncu CSVs:

    T(k, m) = eta(k) * max_r D_r(k, m)          [SM cycles]
    wall    = T(k, m) / (N_SM(m) is already in D) / f(m)

with, per kernel k,

    D_shared = sum_i cost_i(m) * N_i(k)  / N_SM(m)
               cost = FLOP_per_instruction / (FLOP per SM-clock of the pipe)
               i.e. DMMA.8x8x4 -> 4.0 cycles on all three machines,
               CUDA-core FP64 -> 0.50 (GB200, H100) / 1.00 (A100)
    D_dram   = bytes(k)       / (DRAM B per SM-clock of m * N_SM(m))
    D_lsu    = LSU wavefronts / N_SM(m)          (one wavefront per SM-clock)
    D_issue  = instructions   / (4 * N_SM(m))
    D_fma    = FMA pipe inst  / (4 * N_SM(m))
    D_alu    = ALU pipe inst  / (2 * N_SM(m))
    D_l2     = L2 bytes       / (L2 B per SM-clock * N_SM(m))

Everything but D_shared and D_dram is machine-invariant per SM-clock; that is
an assumption of the model, not a fit (section 4 of the report).  eta(k) is
measured on the calibration machine and carried to the target as a kernel
property: one parameter per kernel, no machine dependence.

WHAT THE TARGET MACHINE CONTRIBUTES: its spec sheet only (SM count, boost
clock, the two pipe throughputs, HBM bandwidth).  No target timing, no target
counter, ever enters a prediction.  --audit prints the provenance of every
input so this can be checked.

Usage:
    python3 analysis/subpipe_predict.py --logo            # all three directions
    python3 analysis/subpipe_predict.py --src GB200 --tgt H100 --order 511 -v
"""

import argparse
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Machines.  Spec sheet only.  fp64_tc / fp64_cc are FP64 FLOP per SM-clock,
# dram_B is DRAM bytes per SM-clock; both are peak/(N_SM*f).  Sources are in
# reports/subpipe_transfer_model.md section 1 and reports/h100_report.md 2.
# ---------------------------------------------------------------------------
# `arch` is the -arch= target the binary that runs on the machine is built for.
# It is what decides how a requested MMA shape lowers to hardware instructions
# (MMA_EXPANSION below); `dmma_shapes` is derived from that, not listed here.
MACHINES = {
    "GB200": dict(sm=152, ghz=2.062, fp64_tc=127.9, fp64_cc=127.9, dram_B=25.21,
                  dram_TBs=7.9, arch="sm_100"),
    "A100":  dict(sm=108, ghz=1.410, fp64_tc=128.1, fp64_cc=63.7,  dram_B=10.21,
                  dram_TBs=1.555, arch="sm_80"),
    "H100":  dict(sm=132, ghz=1.980, fp64_tc=256.0, fp64_cc=128.2, dram_B=9.17,
                  dram_TBs=2.396, arch="sm_90"),
}

# Sub-pipe sweeps, all at CutlassMmaShape 8x8x4.  More than one directory per
# machine: the original all-order sweep, then the extensions (p >= 767, and the
# paths the first sweep did not cover).  Searched in order; the first directory
# that has the file wins.
# The lowgemm directories come FIRST: they are the re-take of the unfused GEMM
# paths below p=63 with flux_all in the kernel filter, and they supersede the
# same cells in the original sweeps (section 5Y).
SWEEPS = {
    "GB200": ["subpipe_lowgemm_96222/ncu", "subpipe_orders_92904/ncu",
              "subpipe_hi_94874/ncu", "subpipe_paths_94875/ncu",
              # Last, so it only ever supplies a file the four above lack:
              # (GB200, 767, GEMM_CUTE), which subpipe_hi_94874 did not take
              # and which is the only calibration source for that cell --
              # A100 cannot hold p=767 (40 GB).  Slurm 106536.
              "subpipe_hi_shape_106536/ncu"],
    "A100":  ["wisteria/a100_subpipe_lowgemm_9579761/ncu",
              "wisteria/a100_subpipe_orders_9569112/ncu",
              "wisteria/a100_subpipe_paths_*/ncu"],
    "H100":  ["tsubame/h100_subpipe_lowgemm_8631948/ncu",
              "tsubame/h100_subpipe_orders_*/ncu",
              "tsubame/h100_subpipe_paths_*/ncu",
              # Last, so it only ever supplies a file the three above lack:
              # the p=511 GEMM_CUTE sweep at 16x8x4, which is the only
              # GEMM_CUTE measurement at H100's production shape (SGE 8621412).
              # Verified not to move any 8x8x4 number (V1 item 4).
              "tsubame/h100_subpipe_shape_*/ncu",
              # Also last, for the same reason: (H100, 767, GEMM_CUTE) at
              # 8x8x4 exists only in the repeat of the gaps job (SGE
              # 8631255), so without this entry the cell whose GB200
              # calibration Slurm 106536 supplies has no target to score
              # against at 8x8x4.
              "tsubame/h100_subpipe_repeat_8631255/ncu"],
}


def sweep_dirs(machine):
    """Entries may be globs, so a sweep that comes back from another machine is
    picked up by dropping the directory in place, without editing this file."""
    out = []
    for d in SWEEPS[machine]:
        out += sorted(x for x in glob.glob(os.path.join(ROOT, d))
                      if os.path.isdir(x))
    return out

# Machine-invariant capacities, per SM per SM-clock (section 4).
CAP_ISSUE = 4.0        # issue slots
CAP_LSU_WF = 1.0       # LSU data-pipe wavefronts
CAP_FMA = 4.0          # FMA pipe instructions
CAP_ALU = 2.0          # ALU pipe instructions
CAP_L2_B = 64.0        # L2 bytes per SM-clock (calibrated below, see --audit)

STALLS = ["long_scoreboard", "short_scoreboard", "math_pipe_throttle",
          "mio_throttle", "lg_throttle", "tex_throttle", "barrier", "membar",
          "wait", "drain", "no_instruction", "not_selected"]

METRICS = {
    "dur": "gpu__time_duration.sum",
    "freq": "sm__cycles_elapsed.avg.per_second",
    "inst": "sm__inst_executed.sum",
    "tensor": "sm__inst_executed_pipe_tensor.sum",
    "fp64": "sm__inst_executed_pipe_fp64.sum",
    "alu": "sm__inst_executed_pipe_alu.sum",
    "fma": "sm__inst_executed_pipe_fma.sum",
    "lsu_wf": "l1tex__data_pipe_lsu_wavefronts.sum",
    "dram_B": "dram__bytes.sum",
    "l2_B": "lts__t_bytes.sum",
    "warps_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
}
# Group B of the sweep: the issue-stall breakdown, which is what the second
# equation (latency-bound kernels, section 5H) transfers.
METRICS.update({"st_" + s_: "smsp__average_warps_issue_stalled_%s"
                            "_per_issue_active.ratio" % s_ for s_ in STALLS})


def role_of(name):
    """Canonical kernel role, stable across machines.

    Our own kernels keep their symbol name (identical source on all three
    machines).  cuBLAS does not: the library picks a machine-specific kernel,
    so its launches are collapsed into one role per instruction class -- the
    tensor-op family and the SIMT family (which p <= 15 also gets, report
    section 5K.3).
    """
    if "volume_flux" in name:
        return "volume_flux"
    if "separable_lift" in name:
        return "separable_lift"
    if "elembnd" in name:
        return "elembnd_flux"
    if "gemmSN" in name or "_simt_" in name:
        return "cublas_gemm_simt"
    if ("d884gemm" in name or re.search(r"sm\d+_xmma_gemm", name)
            or "_tensorop_" in name or "dgemm" in name):
        return "cublas_gemm"
    if "GemmBatchedDqdtAssembly" in name:
        return "cutlass_assembly"
    if "GemmBatchedScaleAdd" in name:
        return "cutlass_scaleadd"
    if "cutlass::gemm::kernel::GemmBatched" in name:
        return "cutlass_mainloop"
    m = re.match(r"(?:void\s+)?([A-Za-z_][A-Za-z0-9_]*)", name.strip())
    base = m.group(1) if m else name[:40]
    # tendency_p255_kernel<(int)0,...> -> one role per direction
    d = re.search(r"kernel<\(int\)(\d)", name)
    return base + ("_" + "xyz"[int(d.group(1))] if d else "")


DMMA_RE = re.compile(r"tensor(\d+)x(\d+)x(\d+)")
# CUTLASS spells the instruction shape as the GemmShape inside arch::Mma.
CUTLASS_MMA_RE = re.compile(
    r"arch::Mma<cutlass::gemm::GemmShape<\(int\)(\d+), \(int\)(\d+), \(int\)(\d+)>")


def dmma_flop_of(name):
    """FLOP per DMMA instruction, from the kernel NAME of the build that runs.

    The name says which MMA shape the SASS contains (cuBLAS advertises it,
    e.g. sm90_xmma_..._tensor16x8x8_...; the cutlass_80_tensorop_d884gemm
    family and every hand-written kernel here are m8n8k4).

    This reads a NAME, not a timing -- but for the target machine the name
    comes out of the target's own ncu output, which is a target measurement
    all the same.  See static_dmma_flop() and SHAPE_SRC: the default path asks
    the static table first and only falls back here for cuBLAS.
    """
    m = CUTLASS_MMA_RE.search(name) or DMMA_RE.search(name)
    if m:
        a, b, c = (int(x) for x in m.groups())
        return 2.0 * a * b * c
    if "d884gemm" in name:
        return 2.0 * 8 * 8 * 4
    return 2.0 * 8 * 8 * 4          # CutlassMmaShape = 8x8x4 everywhere else


# ---------------------------------------------------------------------------
# U4: the MMA shape as a STATIC property of the cross-compiled binary.
#
# The model's contract (docstring at the top of this file) is that the target
# machine contributes its spec sheet and the instruction shape of the binary
# that will run on it -- and nothing measured.  Until 2026-09-12 the predictor
# honoured the first half and broke the second: it read `dmma_flop` out of
# `obs`, i.e. out of the TARGET's ncu sweep.  Where the shape differs between
# calibration and target that is a hidden target measurement.
#
# What a binary actually tells you, with no GPU and no run:
#
#     module load nvhpc && cuobjdump -sass scale-dg_extraction
#
# on the sm_100 build of this tree gives 807 kernels (our own plus every
# CUTLASS instantiation) and 46340 MMA instructions, of which 46340 are
# `DMMA.8x8x4` and none is anything else.  That is the whole of what we
# compile: `CutlassMmaShape` is 8x8x4 and every hand-written kernel issues
# `mma.sync.m8n8k4`.  The same source cross-compiles to sm_80 and sm_90, so
# the shape is a property of the source, not of the machine -- which is why
# the table below is flat and why an override slot exists anyway (a machine
# whose build did differ would need one row here, still no measurement).
#
# cuBLAS is NOT in the binary -- `ldd` shows libcublas.so.13 -- and which
# kernel family it dispatches is a runtime heuristic (report section 5AB.2).
# So its shape is not statically knowable, by construction and not by
# omission.  static_dmma_flop() returns None for it, and SHAPE_SRC decides
# what happens then.
# ---------------------------------------------------------------------------
OWN_MMA_SHAPE = "8x8x4"

# V1 (2026-09-12): how a REQUESTED MMA shape lowers to hardware instructions,
# per -arch target.  Measured here, with no GPU: one kernel per
# `mma.sync.aligned.m*.f64` PTX instruction in one .cu, `nvcc -arch=<t> -cubin`,
# `cuobjdump -sass | grep -c DMMA` (nvhpc 26.3, CUDA 13.1.115, 2026-09-12).
# Reproduces reports/sm90_mma_shape_survey.md section 6 exactly.
#
#     PTX                 sm_80        sm_90              sm_100
#     mma.sync.m8n8k4     DMMA.884 x1  DMMA.8x8x4 x1      DMMA.8x8x4 x1
#     mma.sync.m16n8k4    (not compilable) DMMA.16x8x4 x1 DMMA.8x8x4 x2
#     mma.sync.m16n8k8    (not compilable) DMMA.16x8x8 x1 DMMA.8x8x4 x4
#     mma.sync.m16n8k16   (not compilable) DMMA.16x8x16 x1 DMMA.8x8x4 x8
#
# sm_80 refuses the three wide shapes at ptxas time ("Feature '.m16n8k4 with
# double types' requires .target sm_90 or higher"), which is the static form of
# "A100 has no shape knob" (section 5X.3 had it as a device trap at run time).
#
# This is the expansion rule the paper's instruction-priced sub-pipe section
# asserts, and it is a property of ptxas and the target, not of the machine:
# readable from a cross-compile, which is why it belongs in a static table.
MMA_EXPANSION = {
    ("sm_80", "8x8x4"): 1,
    ("sm_90", "8x8x4"): 1, ("sm_90", "16x8x4"): 1,
    ("sm_90", "16x8x8"): 1, ("sm_90", "16x8x16"): 1,
    ("sm_100", "8x8x4"): 1, ("sm_100", "16x8x4"): 2,
    ("sm_100", "16x8x8"): 4, ("sm_100", "16x8x16"): 8,
}

# Which roles the namelist entry CutlassMmaShape actually moves.  Measured from
# the 16x8x4 sweeps (tsubame/h100_subpipe_orders_16x8x4 and
# h100_subpipe_shape_8621412): every `cutlass*` role goes 512 -> 1024 FLOP per
# instruction with exactly half the tensor count, at every order from p=7 to
# p=575 and for both GEMM_FUSED and GEMM_CUTE -- including the x GEMM, whose
# XAdopted* aliases in cuda_cutlass_gemm_fused.cu carry the shape since the
# commit that added them.  Our own kernels (volume_flux, elembnd_flux,
# separable_lift, flux_all, the fused tendency kernels) stay at m8n8k4, which
# is hard-coded in their source.  cuBLAS ignores CutlassMmaShape entirely, as
# it must: it is a different library.
CUTLASS_ROLE_PREFIX = "cutlass"

# (machine, role base) -> shape string.  Overrides for a build whose own
# kernels differ from OWN_MMA_SHAPE; empty because cuobjdump says none does
# (sm_100 build of this tree: 807 kernels, 46340 MMA, all DMMA.8x8x4).
STATIC_MMA_SHAPE = {}


def parse_shape(shape):
    m, n, k = (int(x) for x in shape.split("x"))
    return m, n, k


def static_dmma_flop(machine, role, req_shape=OWN_MMA_SHAPE):
    """FLOP per MMA instruction from the binary alone, or None if not knowable.

    `req_shape` is the CutlassMmaShape the target will be run with.  It reaches
    the hardware through MMA_EXPANSION: the same requested 16x8x4 is one
    1024-FLOP DMMA.16x8x4 on sm_90 and two 512-FLOP DMMA.8x8x4 on sm_100, and
    `demands()` prices per hardware instruction, so that is what is returned.

    Ignoring `req_shape` -- which this function did between 2026-09-12's U4 and
    V1 of the same day -- made `--shape static` shape-blind: it priced a
    16x8x4 run of H100 as if it were 8x8x4 and turned the section 5X ranking
    (predicted GEMM_FUSED/FUSED_TC at p=511: 0.710 against a measured 0.857)
    into 1.096, i.e. the wrong side of 1.

    None means "the library picks at runtime" -- today that is exactly the
    cuBLAS roles.  It is a stated limit of the transfer, not a gap to paper
    over; --shape controls whether it is papered over.
    """
    base = role.split("#")[0]
    if base.startswith("cublas"):
        return None
    shape = STATIC_MMA_SHAPE.get((machine, base), OWN_MMA_SHAPE)
    if base.startswith(CUTLASS_ROLE_PREFIX):
        shape = req_shape
    n_inst = MMA_EXPANSION.get((MACHINES[machine]["arch"], shape))
    if n_inst is None:
        raise ValueError("%s (%s) cannot issue CutlassMmaShape %s"
                         % (machine, MACHINES[machine]["arch"], shape))
    m, n, k = parse_shape(shape)
    return 2.0 * m * n * k / n_inst


# How the target's MMA shape is obtained.
#   "static"      static table, falling back to the target's observed kernel
#                 name for the roles the table cannot know (cuBLAS).  Default:
#                 numerically identical to the pre-2026-09-12 behaviour,
#                 because the table and the observation agree everywhere the
#                 table has an answer.
#   "static-only" static table only; no fallback, so a cuBLAS role is priced
#                 with the calibration machine's shape.  This is the version
#                 that actually honours the published contract.  Scored in
#                 report section 5AC.4.
#   "target"      the old behaviour, observed name first.  Kept to show that
#                 "static" reproduces it exactly.
SHAPE_SRC = ["static"]


def target_dmma_flop(role, obs, tgt, req_shape=OWN_MMA_SHAPE):
    """FLOP per MMA instruction to price `role` with on machine `tgt`."""
    if SHAPE_SRC[0] == "target":
        return obs.get(role, {}).get("dmma_flop") if obs else None
    st = static_dmma_flop(tgt, role, req_shape)
    if st is not None:
        return st
    if SHAPE_SRC[0] == "static-only":
        return None                 # -> demands() uses the calibration shape
    return obs.get(role, {}).get("dmma_flop") if obs else None


def read_csv(path):
    """-> {launch_id: (role, {metric: value})}"""
    with open(path, errors="replace") as fh:
        rows = list(csv.reader(fh))
    hdr_i = next((i for i, r in enumerate(rows) if r and r[0] == "ID"), None)
    if hdr_i is None:
        return {}
    hdr = rows[hdr_i]
    ix = {k: hdr.index(k) for k in ("ID", "Kernel Name", "Metric Name", "Metric Unit",
                                    "Metric Value")}
    want = {v: k for k, v in METRICS.items()}
    out = defaultdict(lambda: [None, {}])
    for r in rows[hdr_i + 1:]:
        if len(r) != len(hdr):
            continue
        mn = r[ix["Metric Name"]]
        if mn not in want:
            continue
        try:
            val = float(r[ix["Metric Value"]].replace(",", ""))
        except ValueError:
            continue
        unit = r[ix["Metric Unit"]]
        if want[mn] == "dur":                       # normalise to ns
            val *= {"nsecond": 1.0, "usecond": 1e3, "msecond": 1e6,
                    "second": 1e9}.get(unit, 1.0)
        if want[mn] == "freq":
            val *= {"cycle/nsecond": 1e9, "cycle/usecond": 1e6,
                    "cycle/msecond": 1e3, "cycle/second": 1.0}.get(unit, 1.0)
        e = out[r[ix["ID"]]]
        e[0] = role_of(r[ix["Kernel Name"]])
        e[1]["_name"] = r[ix["Kernel Name"]]
        e[1][want[mn]] = val
    return out


TABLE = os.path.join(ROOT, "analysis", "subpipe_inputs.csv")
TABLE_COLS = (["machine", "order", "path", "role", "n", "mult", "dmma_flop"]
              + list(METRICS))
_TABLE_CACHE = {}
USE_TABLE = [False]


def load_table():
    """The distilled inputs, which are what is committed (the raw ncu CSVs are
    100 MB).  One row per (machine, order, path, kernel role)."""
    if _TABLE_CACHE:
        return _TABLE_CACHE
    with open(TABLE) as fh:
        for r in csv.DictReader(fh):
            key = (r["machine"], int(r["order"]), r["path"])
            agg = {"n": int(r["n"]), "mult": int(r["mult"]),
                   "dmma_flop": float(r["dmma_flop"])}
            for k in METRICS:
                agg[k] = float(r[k])
            _TABLE_CACHE.setdefault(key, {})[r["role"]] = agg
    return _TABLE_CACHE


def load_case(machine, order, path_name, shape="8x8x4"):
    if shape == "8x8x4" and (USE_TABLE[0] or not sweep_dirs(machine)):
        return load_table().get((machine, order, path_name), {})
    return load_case_raw(machine, order, path_name, shape)


def launch_multiplicity(seq):
    """Launches of each role in one tendency call, from the launch order.

    `seq` is the roles in launch order.  A tendency call is one period of that
    sequence, so find the shortest period that the whole sequence agrees with
    (allowing the window to clip a partial call at either end) and count each
    role inside it.  Falls back to one launch per role if nothing is periodic.
    """
    n = len(seq)
    for period in range(1, n // 2 + 1):
        if all(seq[i] == seq[i % period] for i in range(n)):
            out = defaultdict(int)
            for r in seq[:period]:
                out[r] += 1
            return dict(out)
    # No exact period (the window can start mid-call): try every rotation of a
    # candidate period and take the first that fits the interior.
    for period in range(1, n // 2 + 1):
        for off in range(period):
            if all(seq[i] == seq[off + (i - off) % period]
                   for i in range(off, n)):
                out = defaultdict(int)
                for r in seq[off:off + period]:
                    out[r] += 1
                if len(out) == len(set(seq)):
                    return dict(out)
    return {r: 1 for r in set(seq)}


def split_occurrences(per_role, mult):
    """role -> role#i for the i-th launch of that role inside one call."""
    out = {}
    for role, lst in per_role.items():
        m = mult.get(role, 1)
        if m <= 1:
            out[role] = lst
            continue
        for i in range(m):
            sub = lst[i::m]
            if sub:
                out["%s#%d" % (role, i + 1)] = sub
    return out


def load_case_raw(machine, order, path_name, shape="8x8x4"):
    """Per-role, per-launch means for one (machine, order, path).

    Returns {role: {"n": launches, "mult": launches per tendency call, metrics...}}
    """
    files = []
    for d in sweep_dirs(machine):
        f = sorted(glob.glob(os.path.join(d, "p%d_%s_%s_none_r*_a.csv"
                                          % (order, path_name, shape))))
        # The stall metrics are collected in a second ncu pass (suffix _b), a
        # different set of launches of the same kernels; merged per role below.
        f += sorted(glob.glob(os.path.join(d, "p%d_%s_%s_none_r*_b.csv"
                                           % (order, path_name, shape))))
        if f:
            files = f
            break
    per_role = defaultdict(list)
    stall_role = defaultdict(list)
    order_seq = []
    order_seq_done = [False]
    for f in files:
        b_pass = f.endswith("_b.csv")
        for _, (role, mets) in sorted(read_csv(f).items(),
                                      key=lambda kv: int(kv[0])):
            if role and "dur" in mets:
                (stall_role if b_pass else per_role)[role].append(mets)
                if not b_pass and not order_seq_done[0]:
                    order_seq.append(role)
        if not b_pass and order_seq:
            order_seq_done[0] = True
    if not per_role:
        return {}
    # How many launches of each role make up one tendency call.  Counting them
    # as "count / (launches of the rarest role)" is wrong when the profiling
    # window does not start and end on a call boundary: at p=1023 the GEMM path
    # gave 15 cuBLAS launches against 3 of each peripheral kernel, so the cuBLAS
    # multiplicity came out 5 where the path issues 3, and the per-call total
    # read 300 ms against the 194 ms/stage of p1023_gap_study.md.  Take the
    # period of the launch SEQUENCE instead, which is what a call is.
    # Split a role that appears more than once per call into its occurrences,
    # because they are not the same kernel: the three volume GEMMs are x, y and
    # z, and at p=1023 the plain-GEMM path issues five cuBLAS launches whose
    # durations differ (the period is CCCCCLve).  Taking a median over them and
    # multiplying by the count read 300 ms/call against the 194 ms/stage of
    # p1023_gap_study.md; per occurrence it is right.
    mult = launch_multiplicity(order_seq)
    per_role, stall_role = split_occurrences(per_role, mult), \
        split_occurrences(stall_role, mult)
    mult = {r: 1 for r in per_role}
    out = {}
    for role, lst in per_role.items():
        agg = {"n": len(lst), "mult": mult.get(role, 1),
               "dmma_flop": statistics.median([dmma_flop_of(x["_name"])
                                               for x in lst])}
        for k in METRICS:
            src_lst = stall_role.get(role, []) if k.startswith("st_") else lst
            vals = [x[k] for x in src_lst if k in x]
            agg[k] = statistics.median(vals) if vals else 0.0
        out[role] = agg
    return out


def demands(agg, machine, dmma_flop_tgt=None):
    """Resource demands in SM cycles for one kernel launch on `machine`.

    `dmma_flop_tgt` is the FLOP per DMMA instruction of the code that will run
    on the target (read from the target build's kernel name, never from a
    target measurement).  It differs from the calibration machine only where
    the library picks a different kernel, i.e. cuBLAS: sm_80/sm_100 issue
    m8n8k4, sm_90 issues m16n8k8.  The instruction count is rescaled by the
    FLOP ratio, so the arithmetic is held fixed and only the pricing moves.
    """
    m = MACHINES[machine]
    n_sm = m["sm"]
    flop_cal = agg.get("dmma_flop", 512.0)
    flop_tgt = dmma_flop_tgt or flop_cal
    n_tensor = agg["tensor"] * (flop_cal / flop_tgt)
    # Shared fp64+tensor pipe, priced per instruction (section 2).
    # cost = max(4, FLOP_instr / C_tc,m).  A DMMA instruction occupies the
    # shared pipe for at least 4 cycles whatever the machine (measured
    # 3.97-4.03 on all three, report section 2 and 5A.1); H100's 256 FLOP per
    # SM-clock is reachable only through the wider shapes it has and 8x8x4
    # does not get cheaper there.  This is the one place the machine enters
    # other than through capacities, and it is what the GB200+A100 pair is
    # rank-deficient in (report section 1).
    cost_dmma = max(4.0, flop_tgt / m["fp64_tc"])
    cost_fp64 = 64.0 / m["fp64_cc"]             # warp-wide FP64 FMA = 32*2 FLOP
    d = {
        "shared": (cost_dmma * n_tensor + cost_fp64 * agg["fp64"]) / n_sm,
        "dram":   agg["dram_B"] / (m["dram_B"] * n_sm),
        "lsu_wf": agg["lsu_wf"] / (CAP_LSU_WF * n_sm),
        "issue":  agg["inst"] / (CAP_ISSUE * n_sm),
        "fma":    agg["fma"] / (CAP_FMA * n_sm),
        "alu":    agg["alu"] / (CAP_ALU * n_sm),
        "l2":     agg["l2_B"] / (CAP_L2_B * n_sm),
    }
    return d


def cycles_measured(agg, machine):
    f = agg.get("freq") or MACHINES[machine]["ghz"] * 1e9
    return agg["dur"] * 1e-9 * f


def eta_of(agg, machine):
    d = demands(agg, machine)
    top = max(d.values())
    return (cycles_measured(agg, machine) / top if top > 0 else float("nan"),
            max(d, key=d.get))


ETA_RULE = ["closest"]
def stall_cycles(agg, src, tgt):
    """Second equation (section 5H): T = N_inst (1 + sum S) / (4 W) SM cycles.

    An identity on the machine it is measured on; transferring it means
    transferring each stall class.  Measured laws (section 5M):

      long_scoreboard  a DRAM queue -> scales with DRAM bytes per SM-clock
                       (57.4% of the weighted stall; measured 2.40 / 2.61
                       against 2.47 / 2.75, section 5O)
      math_pipe        machine-invariant (13.1%; measured 1.15 / 0.84).  NOT
                       the shared-pipe demand ratio, which is 2.01 on A100 and
                       would double-count what the roof term already carries
      wait             fixed instruction latencies, invariant in CYCLES
                       (10.9%; measured 0.97 / 0.95)
      short_scoreboard, not_selected
                       shared-memory latency, fixed in NANOSECONDS, so it
                       scales with the clock (measured 0.57 / 0.97 and
                       0.73 / 0.97 against 0.68 / 0.96)
      mio / lg_throttle / barrier / drain
                       no law found; 12.9% of the weighted stall and the
                       largest remaining hole in the second equation
                       (section 5O.2)

    N_inst and W are kernel properties and carry (measured 0.92-1.02 and
    1.02-1.06 across the three machines).
    """
    W = agg["warps_pct"] / 100.0 * 16.0          # warps per scheduler
    if W <= 0:
        return float("inf")
    bw = MACHINES[src]["dram_B"] / MACHINES[tgt]["dram_B"]
    fr = MACHINES[tgt]["ghz"] / MACHINES[src]["ghz"]
    rate = ((MACHINES[src]["sm"] * MACHINES[src]["ghz"])
            / (MACHINES[tgt]["sm"] * MACHINES[tgt]["ghz"]))
    sh_s = demands(agg, src)["shared"]
    sh_t = demands(agg, tgt)["shared"]
    if USE_PROBE[0]:
        long_law = (PROBE_L0[tgt] / PROBE_L0[src]) * QUEUE_K * bw
    else:
        long_law = UNLOADED_SHARE * fr + (1 - UNLOADED_SHARE) * bw
    if CLAMP_Q[0]:
        # REJECTED, kept because the reason is instructive (section 5P.2).
        # The queue inflation Q has a floor of 1, so it is tempting to floor
        # the transfer RATIO at the clock ratio.  That is wrong and measures
        # worse (the GB200 hold-out goes 41% -> 105% at p <= 31): the floor is
        # on each machine's absolute Q, not on the ratio, and a ratio below 1
        # is exactly right when the source machine is the queued one and the
        # target is not.  Separating them needs Q on the calibration machine,
        # which the two-clock sweep measures and nothing else does.
        long_law = max(long_law, fr)
    law = {"long_scoreboard": long_law,
           "math_pipe_throttle": 1.0,
           "short_scoreboard": fr, "not_selected": fr,
           "mio_throttle": rate, "lg_throttle": rate, "drain": rate}
    tot = 1.0 + sum(agg.get("st_" + c, 0.0) * law.get(c, 1.0) for c in STALLS)
    return agg["inst"] / MACHINES[tgt]["sm"] * tot / (4.0 * W)


# Share of the exposed long-scoreboard wait that is unloaded latency rather
# than queueing, on the calibration machine.  Two machines give the same value
# to within 2 points (0.061 from A100, 0.083 from H100), which is one degree
# of freedom of validation, not none -- but it is the one fitted number in the
# model and section 5M says so.
UNLOADED_SHARE = 0.07

# Unloaded global-memory latency, SM cycles, from jobs/global_latency_probe.cu.
#     S_long ratio = (L0 ratio in SM cycles) x QUEUE_K x (DRAM B/SM-clk ratio)
# QUEUE_K is fitted on two points that agree to 5% (1.165 from A100, 1.220 from
# H100, on the fixed-probe L0 below; 1.169 / 1.230 on the original probe), and
# this form reproduces both measured S_long ratios to 1%.
#
# These went to co-run values 462.5 / 355.3 / 441.5 for one commit and came
# back: the "co-run latency" was the chase hitting in L2 on its own lines, not a
# property of a busy machine (section 5R).
#
# All three are now re-taken with the fixed probe, which evicts L2 and starts
# each chase at an address it has not visited (section 5R.7): 450.3 / 485.2 /
# 374.7 ns, against 444.0 / 477.0 / 366.4 from the original version -- +1.4 to
# +2.3%, the residual reuse its 64-step warm-up left behind.  The A100 clock
# caveat of 5Q.6 closed with it: this wait is DRAM nanoseconds and barely
# depends on the SM clock.
PROBE_L0 = {"GB200": 928.5, "A100": 684.1, "H100": 741.9}
QUEUE_K = 1.20

# Where the roof model stops applying, on the calibration machine's own eta.
# Section 5E.3 measured the error bands at this boundary; it is not fitted to
# any hold-out.
AUTO_ETA = 1.2

ETA_RULE = ["closest"]
MODEL = ["roof"]
USE_PROBE = [False]
CLAMP_Q = [False]
# FUSED_TC first: it is the denominator of the ranking ratios below.
# PATHS is the DISTILL list: what --distill pulls out of the raw ncu sweeps and
# writes into analysis/subpipe_inputs.csv.  GEMM_OZAKI1 stays in it because the
# Ozaki appendix uses those rows.
PATHS = ["FUSED_TC", "GEMM_FUSED", "GEMM", "FUSED", "FUSED_DFMA", "GEMM_CUTE",
         "GEMM_OZAKI1"]

# U2: what the leave-one-GPU-out score and --dump-logo actually range over.
# GEMM_OZAKI1 is NOT transferred -- section 5U.3 took it out of scope and
# section 5AB.2 excluded it again when scoring -- but it stayed in the scored
# list until 2026-09-12 and contributed 23 of the 161 dumped rows.  It does not
# belong there on its own terms either: its ncu capture is a single
# `cublas_gemm` role at mult=1, so the row is one kernel, not a path total
# (GB200 p=511 reads 106.2 us against the 174511.4 us the appendix measures for
# that path -- three orders of magnitude, because the split and sum kernels
# were never captured).  Scoring it was scoring a kernel against a path.
NO_TRANSFER = {"GEMM_OZAKI1"}
LOGO_PATHS = [p_ for p_ in PATHS if p_ not in NO_TRANSFER]
ORDERS = [7, 15, 31, 63, 127, 255, 511, 575, 767, 1023]
# Each pair isolates one axis: fusion (CUTE vs FUSED), the mma instruction on
# an iso-schedule (DFMA vs TC), the library under one schedule (CUTE vs GEMM),
# the mma fragment layout against natural order on CUDA cores (DFMA vs FUSED),
# and CUDA core against tensor core (FUSED vs TC).
RATIOS = [("GEMM_FUSED", "FUSED_TC"), ("GEMM", "FUSED_TC"),
          ("FUSED_DFMA", "FUSED_TC"), ("GEMM_CUTE", "GEMM_FUSED"),
          ("GEMM_CUTE", "GEMM"), ("FUSED_DFMA", "FUSED"),
          ("FUSED", "FUSED_TC")]


def predict_case(srcs, tgt, order, path_name, verbose=False, tgt_shape="8x8x4"):
    """Predict per-tendency-call device time on `tgt` from calibration `srcs`.

    Returns (predicted_us, measured_us or None, rows, ledger_ok).

    `ledger_ok` is False when the calibration machines' kernel ledger for this
    path is not the same set of roles as the target's.

    U3: pred and meas are summed over the SAME set of roles.  Until 2026-09-12
    they were not: `roles` was the union over the calibration machines and
    pred summed all of it, while meas (below) only added the target launches
    whose role name happened to match.  So a role the target does not have was
    predicted and not measured, and a role only the target has was measured by
    neither -- the error of such a cell is an artefact of the mismatch, not of
    the model.  Section 5AB.1 found one instance of this (H100's
    sm80_xmma_gemm) and fixed the role name; the general form is here.
    Intersecting is the conservative repair: it scores the part of the path
    both ledgers agree on, and `ledger_ok` marks the cell so a consumer can
    drop it entirely.  It is NOT a claim that the dropped kernels are free --
    which is exactly why the flag exists.

    When the target has no sweep at all (the balance axis builds pseudo
    machines with SWEEPS = []), obs is empty and there is nothing to intersect
    with; the union is used and `ledger_ok` is True vacuously.
    """
    cal = {s: load_case(s, order, path_name) for s in srcs}
    cal = {s: c for s, c in cal.items() if c}
    if not cal:
        return None, None, [], True
    obs = load_case(tgt, order, path_name, shape=tgt_shape)
    roles = set()
    for c in cal.values():
        roles |= set(c)
    ledger_ok = True
    if obs:
        ledger_ok = (roles == set(obs))
        roles &= set(obs)
    pred_ns = 0.0
    meas_ns = 0.0
    rows = []
    for role in sorted(roles):
        # eta rule for >1 calibration machine: take the machine whose kernel
        # sits closest to its roof (eta nearest 1) -- the more constrained
        # machine, per section 5I.3.  Decidable from calibration data alone.
        cands = []
        for s, c in cal.items():
            if role in c:
                e, r = eta_of(c[role], s)
                cands.append((abs(e - 1.0), s, e, r, c[role]))
        if not cands:
            continue
        rule = ETA_RULE[0]
        if rule == "min":
            _, src, eta, roof_src, agg = min(cands, key=lambda t: t[2])
        elif rule == "max":
            _, src, eta, roof_src, agg = max(cands, key=lambda t: t[2])
        else:
            _, src, eta, roof_src, agg = min(cands)
            if rule == "mean":
                eta = sum(t[2] for t in cands) / len(cands)
        d = demands(agg, tgt, dmma_flop_tgt=target_dmma_flop(role, obs, tgt,
                                                             tgt_shape))
        roof = max(d, key=d.get)
        cyc = eta * d[roof]
        if MODEL[0] != "roof" and agg.get("warps_pct"):
            st_cyc = stall_cycles(agg, src, tgt)
            if MODEL[0] == "stall":
                cyc = st_cyc
            elif MODEL[0] == "max":
                cyc = max(d[roof], st_cyc)
            else:                                    # auto
                cyc = cyc if eta <= AUTO_ETA else max(d[roof], st_cyc)
        ns = cyc / (MACHINES[tgt]["ghz"] * 1e9) * 1e9 * agg["mult"]
        pred_ns += ns
        o = obs.get(role)
        ons = o["dur"] * o["mult"] if o else 0.0
        meas_ns += ons
        rows.append(dict(role=role, src=src, eta=eta, roof_src=roof_src,
                         roof_tgt=roof, pred_us=ns / 1e3,
                         meas_us=ons / 1e3 if o else None,
                         roof_obs=(eta_of(o, tgt)[1] if o else None)))
    return pred_ns / 1e3, (meas_ns / 1e3 if obs else None), rows, ledger_ok


# Bins of the eta stratification (report section 5AC.6, paper Table
# "eta-strata").  eta is measured on the CALIBRATION machine, so the bin a
# kernel falls in is known before the target is touched -- that is the whole
# point of the table.
ETA_BINS = [("1.0-1.2", 1.0, 1.2), ("1.2-1.6", 1.2, 1.6),
            ("1.6-2.5", 1.6, 2.5), (">2.5", 2.5, float("inf"))]


def pct90(vals):
    """90th percentile, NEAREST-RANK: the ceil(0.9 n)-th smallest, 1-based.

    Fixed here because the paper's table was built with
    `sorted(v)[round(0.9 * (n - 1))]`, which is not any standard convention
    and degenerates at small n: at n = 8 it returns the 7th of 8 values, i.e.
    the 81st percentile.  Nearest-rank is defined for every n and needs no
    interpolation.  Note that the two highest bins hold 5 to 12 kernels, where
    a 90th percentile carries almost no information whatever the convention --
    report the n beside it (section 5AC.6).
    """
    v = sorted(vals)
    import math
    return v[max(0, math.ceil(0.9 * len(v)) - 1)] if v else None


def eta_strata(dirs, only_ledger_ok=False):
    """Per-kernel transfer errors bucketed by eta on the calibration machine.

    One row per (hold-out direction, order, path, kernel role) that the target
    actually measured.  `only_ledger_ok` drops the cells whose calibration and
    target kernel ledgers disagree -- which is also exactly what empties the
    `> 2.5` bin, because that bin is entirely tiny low-order cuBLAS launches
    (section 5AC.6c).
    """
    buck = defaultdict(list)
    for srcs, tgt in dirs:
        for p_ in ORDERS:
            for path_name in LOGO_PATHS:
                pr, me, rows, led = predict_case(srcs, tgt, p_, path_name)
                if pr is None or not me or (only_ledger_ok and not led):
                    continue
                for r in rows:
                    if not r["meas_us"] or not r["pred_us"]:
                        continue
                    e = abs(r["pred_us"] / r["meas_us"] - 1.0) * 100.0
                    for nm, a, b in ETA_BINS:
                        if a <= r["eta"] < b:
                            buck[nm].append(e)
                            break
    return buck


def print_eta_strata(dirs):
    for only in (False, True):
        b = eta_strata(dirs, only)
        tot = sum(len(v) for v in b.values())
        print("\n  eta strata (%s): n=%d"
              % ("ledger_ok cells only" if only else "all cells", tot))
        print("    %-8s %5s %9s %9s" % ("eta", "n", "median", "p90"))
        for nm, _, _ in ETA_BINS:
            v = b[nm]
            if not v:
                print("    %-8s %5d %9s %9s" % (nm, 0, "--", "--"))
                continue
            print("    %-8s %5d %8.1f%% %8.1f%%"
                  % (nm, len(v), statistics.median(v), pct90(v)))


def shape_check(dirs, shapes=("8x8x4", "16x8x4")):
    """V1 item 3: is `--shape static` really equal to `--shape target`?

    The 2026-09-12 U4 commit claimed "static reproduces target to the byte".
    That check only ever ran at CutlassMmaShape 8x8x4, because --dump-logo
    passes nothing else -- and `static_dmma_flop` ignored the requested shape,
    so the shape axis was exactly the axis the check did not walk.  This walks
    it: per role, the static table's FLOP per hardware MMA instruction against
    the one read out of the target's own ncu kernel name, for every
    (direction, order, path) that has a sweep at each shape.

    Disagreements are the statically unknowable quantities, printed by role.
    """
    print("\n=== shape equivalence check: static table vs target's own kernel "
          "name")
    tot = defaultdict(lambda: [0, 0])
    bad = defaultdict(lambda: [0, set()])
    for shape in shapes:
        for srcs, tgt in dirs:
            for p_ in ORDERS:
                for path_name in LOGO_PATHS:
                    obs = load_case(tgt, p_, path_name, shape=shape)
                    if not obs:
                        continue
                    for role, agg in sorted(obs.items()):
                        try:
                            st = static_dmma_flop(tgt, role, shape)
                        except ValueError:
                            st = "unsupported"
                        ob = agg.get("dmma_flop")
                        if st is None:
                            tot[(shape, "cuBLAS: no static answer")][1] += 1
                            continue
                        key = (shape, "static == target")
                        tot[key][1] += 1
                        if st == ob:
                            tot[key][0] += 1
                        else:
                            b = bad[(shape, role.split("#")[0])]
                            b[0] += 1
                            b[1].add("%s->%s/%s p=%d: static %s obs %s"
                                     % ("+".join(srcs), tgt, path_name, p_,
                                        st, ob))
    for (shape, what), (ok, n) in sorted(tot.items()):
        if what.startswith("static =="):
            print("  %-8s %-28s %d / %d roles agree" % (shape, what, ok, n))
        else:
            print("  %-8s %-28s %d roles (priced with the calibration shape "
                  "under --shape static-only)" % (shape, what, n))
    if bad:
        print("  DISAGREEMENTS (statically unknowable):")
        for (shape, base), (n, ex) in sorted(bad.items()):
            print("    %-8s %-22s %d launches, e.g. %s"
                  % (shape, base, n, sorted(ex)[0]))
    else:
        print("  no disagreement on any role the static table answers, at "
              "either shape")
    return not bad


def fmt_err(p, m):
    return "%+6.1f%%" % ((p / m - 1.0) * 100.0) if (p and m) else "    n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", default=None)
    ap.add_argument("--tgt")
    ap.add_argument("--order", type=int)
    ap.add_argument("--logo", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--distill", action="store_true",
                    help="rebuild analysis/subpipe_inputs.csv from the raw ncu CSVs")
    ap.add_argument("--from-table", action="store_true",
                    help="ignore the raw ncu CSVs and use the committed table")
    ap.add_argument("--model", default="roof", choices=["roof", "stall", "max", "auto"],
                    help="roof: eta * max demand (section 4).  stall: the "
                         "second equation (section 5H/5M).  max: the larger of "
                         "the unscaled roof and the second equation.  auto: "
                         "roof where the calibration machine says the kernel is "
                         "near a roof (eta <= AUTO_ETA), the second equation "
                         "where it is not -- decidable from calibration data")
    ap.add_argument("--clamp-q", action="store_true",
                    help="floor the transfer ratio at the clock ratio; measured"
                         " worse, kept as a rejected variant (section 5P.2)")
    ap.add_argument("--l0-probe", action="store_true",
                    help="use the measured unloaded latency (jobs/"
                         "global_latency_probe.cu) in the long-scoreboard law "
                         "instead of the two-point fit; costs a two-minute run "
                         "on the target machine")
    ap.add_argument("--eta-rule", default="closest",
                    choices=["closest", "mean", "min", "max"],
                    help="with two calibration machines, how to combine eta")
    ap.add_argument("--dump-logo",
                    help="with --logo, also write one row per (held-out "
                         "machine, order, path) to this CSV: the schema the "
                         "paper's Figure 9 reads (figs/data/"
                         "logo_path_totals.csv)")
    ap.add_argument("--shape", default="static",
                    choices=["static", "static-only", "target"],
                    help="where the target's MMA shape comes from.  static: "
                         "the cuobjdump table, falling back to the target's "
                         "observed kernel name for cuBLAS only (default; "
                         "reproduces --shape target exactly).  static-only: "
                         "no fallback, so nothing measured on the target "
                         "enters -- the version that honours the published "
                         "contract (section 5AC.4).  target: the "
                         "pre-2026-09-12 behaviour")
    ap.add_argument("--tgt-shape", default="8x8x4",
                    help="the CutlassMmaShape the TARGET is run with.  The "
                         "calibration machines are always read at 8x8x4 (the "
                         "only shape their sweeps have).  H100's production "
                         "setting is 16x8x4 (section 5X); sm_80 cannot issue "
                         "anything but 8x8x4, so A100 raises")
    ap.add_argument("--shape-check", action="store_true",
                    help="compare the static MMA-shape table against the "
                         "shape read out of each target's own ncu kernel name, "
                         "at 8x8x4 AND 16x8x4, and exit.  This is the "
                         "equivalence test --shape static has to pass before "
                         "it may be the default (V1 item 3)")
    ap.add_argument("--eta-strata", action="store_true",
                    help="with --logo, also print the eta stratification of "
                         "the per-kernel transfers (section 5AC.6), both over "
                         "all cells and over the ledger-consistent ones.  p90 "
                         "is nearest-rank; see pct90()")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    USE_TABLE[0] = a.from_table
    SHAPE_SRC[0] = a.shape
    ETA_RULE[0] = a.eta_rule
    MODEL[0] = a.model
    USE_PROBE[0] = a.l0_probe
    CLAMP_Q[0] = a.clamp_q

    if a.distill:
        with open(TABLE, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(TABLE_COLS)
            for m in SWEEPS:
                for o in ORDERS:
                    for pth in PATHS:
                        for role, agg in sorted(load_case_raw(m, o, pth).items()):
                            w.writerow([m, o, pth, role, agg["n"], agg["mult"],
                                        "%.6g" % agg["dmma_flop"]]
                                       + ["%.6g" % agg[k] for k in METRICS])
        print("wrote", TABLE)
        return

    if a.audit:
        print("inputs (target contributes spec sheet only):")
        for m in SWEEPS:
            for d in sweep_dirs(m):
                md = os.path.join(os.path.dirname(d), "metadata.txt")
                tags = ([l.strip() for l in open(md)
                         if l.startswith(("job=", "commit="))]
                        if os.path.exists(md) else ["(no metadata)"])
                print("  %-6s %-44s %s"
                      % (m, os.path.relpath(d, ROOT), " ".join(tags[:2])))
        print("  spec sheet:")
        for m, s in MACHINES.items():
            print("    %-6s SM=%3d f=%.3fGHz tc=%.1f cc=%.1f dramB=%.2f"
                  % (m, s["sm"], s["ghz"], s["fp64_tc"], s["fp64_cc"], s["dram_B"]))
        return

    if a.logo or a.shape_check:
        dirs = [(["A100", "H100"], "GB200"), (["GB200", "H100"], "A100"),
                (["GB200", "A100"], "H100")]
    else:
        dirs = [(a.src or ["GB200"], a.tgt or "H100")]

    if a.shape_check:
        shape_check(dirs)
        return

    dump = []
    for srcs, tgt in dirs:
        print("\n===== calibrate %s -> hold out %s (target CutlassMmaShape "
              "%s) =====" % ("+".join(srcs), tgt, a.tgt_shape))
        hdr = "%-5s %-11s %10s %10s %8s" % ("p", "path", "pred us", "meas us", "err")
        print(hdr)
        errs, roof_hit, roof_n, n_mismatch = [], 0, 0, 0
        shape_warned = False
        flat = {}
        ratios = {}
        for p in ([a.order] if a.order else ORDERS):
            for path_name in LOGO_PATHS:
                try:
                    pr, me, rows, led = predict_case(srcs, tgt, p, path_name,
                                                     a.verbose,
                                                     tgt_shape=a.tgt_shape)
                except ValueError as e:
                    if not shape_warned:
                        print("  skipped: %s" % e)
                        shape_warned = True
                    continue
                if pr is None or not me:
                    continue
                errs.append(pr / me - 1.0)
                flat[(p, path_name)] = (pr, me)
                ratios.setdefault(p, {})[path_name] = (pr, me)
                if not led:
                    n_mismatch += 1
                print("%-5d %-11s %10.1f %10.1f %8s%s"
                      % (p, path_name, pr, me, fmt_err(pr, me),
                         "" if led else "   ledger!"))
                dump.append((tgt, p, path_name, pr, me,
                             (pr / me - 1.0) * 100.0, int(led)))
                for r in rows:
                    if r["roof_obs"]:
                        roof_n += 1
                        roof_hit += (r["roof_tgt"] == r["roof_obs"])
                    if a.verbose:
                        print("      %-18s src=%-5s eta=%.3f %s->%s  pred %8.1f "
                              "meas %8.1f %s"
                              % (r["role"], r["src"], r["eta"], r["roof_src"],
                                 r["roof_tgt"], r["pred_us"], r["meas_us"] or 0,
                                 fmt_err(r["pred_us"], r["meas_us"])))
        if errs:
            ae = sorted(abs(e) for e in errs)
            # statistics.median, i.e. the average of the two middle values when
            # n is even -- the same convention as the per-ratio medians below.
            # It printed ae[n // 2] (the upper median) until 2026-09-11, which
            # differed from the numbers the report quotes in the even-n cells.
            print("  path totals: n=%d  |err| median %.1f%%  max %.1f%%"
                  % (len(ae), 100 * statistics.median(ae), 100 * ae[-1]))
            print("  binding resource: %d / %d kernels" % (roof_hit, roof_n))
            print("  kernel-ledger mismatch (cal roles != tgt roles): %d / %d "
                  "cells" % (n_mismatch, len(ae)))
        # Error by order band: the model declares its own validity domain
        # (report section 5E.3), so report inside and outside it separately.
        band = defaultdict(list)
        for (pp, pa), (pr, me) in flat.items():
            band["p<=31" if pp <= 31 else ("p=63..127" if pp <= 127 else "p>=255")
                 ].append(abs(pr / me - 1.0))
        for k in ("p<=31", "p=63..127", "p>=255"):
            if band[k]:
                v = sorted(band[k])
                print("  |err| %-10s n=%2d median %5.1f%%  max %5.1f%%"
                      % (k, len(v), 100 * statistics.median(v), 100 * v[-1]))
        # The ranking question the paper asks.
        print("  %-5s %10s %10s   %10s %10s" % ("p", "GF/TC pred", "GF/TC meas",
                                                "G/TC pred", "G/TC meas"))
        # The pairs the model is asked to rank.  GEMM_FUSED/FUSED_TC is the
        # one the paper opens with; FUSED_DFMA/FUSED_TC is the mma instruction
        # on an iso-schedule, GEMM_CUTE/GEMM_FUSED the epilogue fusion.
        sign = defaultdict(lambda: [0, 0])
        for num, den in RATIOS:
            rows_ = [(p, r) for p, r in sorted(ratios.items())
                     if num in r and den in r]
            if not rows_:
                continue
            print("  %-24s %10s %10s" % (num + " / " + den, "pred", "meas"))
            errs_ = []
            for p, r in rows_:
                pr = r[num][0] / r[den][0]
                me = r[num][1] / r[den][1]
                ok = (pr > 1) == (me > 1)
                sign[num + "/" + den][0] += ok
                sign[num + "/" + den][1] += 1
                errs_.append(abs(pr / me - 1))
                print("    p=%-6d %10.3f %10.3f   %s" % (p, pr, me,
                                                         "o" if ok else "X"))
            print("    |err| median %.1f%%" % (100 * statistics.median(errs_)))
        print("  sign of ratio-1: "
              + ", ".join("%s %d/%d" % (k, v[0], v[1])
                          for k, v in sorted(sign.items())))

    if a.logo and a.eta_strata:
        print_eta_strata(dirs)

    if a.dump_logo and dump:
        with open(a.dump_logo, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tgt", "order", "path", "pred", "meas", "err",
                        "ledger_ok"])
            for tgt, p, path_name, pr, me, e, led in dump:
                w.writerow([tgt, p, path_name, "%.3f" % pr, "%.3f" % me,
                            "%.2f" % e, led])
        print("\nwrote %s (%d rows)" % (a.dump_logo, len(dump)))


if __name__ == "__main__":
    main()
