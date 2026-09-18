#!/usr/bin/env python3
"""Export the sub-pipe model's per-kernel Nsight Compute inputs for publication.

analysis/subpipe_inputs.csv is what the predictor reads: one row per
(machine, order, path, kernel role) with the twelve counters the model needs
plus the stall breakdown, distilled from the three machines' raw ncu sweeps
(102 MB, local).  A reader who wants to re-derive the model from the counters
themselves needs more than that row: which job and commit the row came from,
which kernel the role stands for, the launch configuration, the shared-pipe
utilisation the section-2 price law was fitted on, and the model's own
derived quantities (the per-resource demands, the binding resource, eta).

This script writes that wider table from the same raw sweeps, with the same
aggregation as --distill (per-role median over the profiled launches, roles
split by occurrence within one tendency call), and checks that every column
the two tables share is identical to analysis/subpipe_inputs.csv.  Counter
columns are named by their ncu metric name so that the README is the metric
list and nothing else.

    python3 analysis/subpipe_inputs_export.py --out FILE.csv [--readme FILE.md]

Only runs where the raw sweeps are (RIKYU); elsewhere the committed
subpipe_inputs.csv is the distilled form.
"""
import argparse
import collections
import csv
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subpipe_predict as sp                                   # noqa: E402

# Group A of jobs/subpipe_ncu.sh, in the order the job requests them.  The
# seven marked "later" were added by the latency job (Slurm 93312,
# reports/subpipe_transfer_model.md section 5H.2) and are absent from the
# first GB200 order sweep (92904), which ran before it; the two *_cmd_read
# counters are also n/a on sm_80 (section 5N.3).  A row without a value
# leaves the column blank rather than 0.
GROUP_A = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg.per_second",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_tensor.sum",
    "sm__inst_executed_pipe_fp64.sum",
    "sm__inst_executed_pipe_alu.sum",
    "sm__inst_executed_pipe_fma.sum",
    "sm__inst_executed_pipe_lsu.sum",
    "sm__inst_executed_pipe_adu.sum",
    "sm__inst_executed_pipe_cbu.sum",
    "sm__inst_executed_pipe_uniform.sum",
    "sm__inst_executed_pipe_xu.sum",
    "smsp__inst_executed.avg.per_cycle_active",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "l1tex__data_pipe_lsu_wavefronts.sum",
    "l1tex__t_sectors.sum",
    "l1tex__t_requests.sum",
    "l1tex__lsuin_requests.sum",
    "l1tex__f_wavefronts.sum",                                      # later
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",               # later
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",               # later
    "sm__inst_executed_pipe_lsu_mem_lg_cmd_read.sum",               # later
    "sm__inst_executed_pipe_lsu_mem_shared_cmd_read.sum",           # later
    "lts__t_sectors_lookup_hit.sum",                                # later
    "lts__t_sectors_lookup_miss.sum",                               # later
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__grid_size",
    "launch__block_size",
]
# Group B: the issue-stall breakdown, a second ncu pass over other launches of
# the same kernels (merged per role, as in --distill).
GROUP_B = ["smsp__average_warps_issue_stalled_%s_per_issue_active.ratio" % s
           for s in sp.STALLS]
STALL_COL = {"smsp__average_warps_issue_stalled_%s_per_issue_active.ratio" % s:
             "st_" + s for s in sp.STALLS}

# Columns of subpipe_inputs.csv that are ncu counters, by metric name.
BASE_OF = {v: k for k, v in sp.METRICS.items()}
# The rest of the base table, in its own order.
BASE_HEAD = ["machine", "order", "path", "role", "n", "mult", "dmma_flop"]

RESOURCES = ["shared", "dram", "lsu_wf", "issue", "fma", "alu", "l2"]


def read_all(path):
    """Like sp.read_csv but keeps EVERY metric, with its unit normalised the
    same way for duration and clock."""
    saved = sp.METRICS
    ext = dict(sp.METRICS)
    for mn in GROUP_A + GROUP_B:
        if mn not in BASE_OF:
            ext[mn] = mn
    sp.METRICS = ext
    try:
        return sp.read_csv(path)
    finally:
        sp.METRICS = saved


def files_for(machine, order, path_name, shape="8x8x4"):
    for d in sp.sweep_dirs(machine):
        f = sorted(glob.glob(os.path.join(d, "p%d_%s_%s_none_r*_a.csv"
                                          % (order, path_name, shape))))
        f += sorted(glob.glob(os.path.join(d, "p%d_%s_%s_none_r*_b.csv"
                                           % (order, path_name, shape))))
        if f:
            return d, f
    return None, []


def metadata(sweep_dir):
    md = os.path.join(os.path.dirname(sweep_dir), "metadata.txt")
    out = {"job": "", "commit": "", "date": ""}
    if os.path.exists(md):
        for line in open(md):
            for k in out:
                if line.startswith(k + "="):
                    out[k] = line.split("=", 1)[1].split()[0].strip()
    return out


def load_case_ext(machine, order, path_name):
    """sp.load_case_raw with every metric kept, plus provenance.  The
    aggregation is copied line for line so the shared columns can be checked
    against the committed table."""
    d, files = files_for(machine, order, path_name)
    if not files:
        return None, {}
    per_role = collections.defaultdict(list)
    stall_role = collections.defaultdict(list)
    order_seq = []
    order_seq_done = [False]
    for f in files:
        b_pass = f.endswith("_b.csv")
        for _, (role, mets) in sorted(read_all(f).items(),
                                      key=lambda kv: int(kv[0])):
            if role and "dur" in mets:
                (stall_role if b_pass else per_role)[role].append(mets)
                if not b_pass and not order_seq_done[0]:
                    order_seq.append(role)
        if not b_pass and order_seq:
            order_seq_done[0] = True
    if not per_role:
        return d, {}
    mult = sp.launch_multiplicity(order_seq)
    per_role = sp.split_occurrences(per_role, mult)
    stall_role = sp.split_occurrences(stall_role, mult)
    out = {}
    for role, lst in per_role.items():
        agg = {"n": len(lst), "mult": 1,
               "dmma_flop": statistics.median([sp.dmma_flop_of(x["_name"])
                                               for x in lst])}
        names = collections.Counter(x["_name"] for x in lst)
        agg["kernel_name"] = names.most_common(1)[0][0]
        agg["kernel_names_seen"] = len(names)
        for mn in GROUP_A + GROUP_B:
            key = BASE_OF.get(mn, mn)
            src = stall_role.get(role, []) if mn in GROUP_B else lst
            vals = [x[key] for x in src if key in x]
            agg[key] = statistics.median(vals) if vals else None
        out[role] = agg
    return d, out


def dmma_shape(flop):
    return {512.0: "8x8x4", 1024.0: "16x8x4", 2048.0: "16x8x8",
            4096.0: "16x8x16"}.get(float(flop), "")


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return "%.6g" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--readme")
    a = ap.parse_args()

    committed = {}
    with open(sp.TABLE) as fh:
        for r in csv.DictReader(fh):
            committed[(r["machine"], r["order"], r["path"], r["role"])] = r

    counter_cols = [mn for mn in GROUP_A if mn not in
                    ("gpu__time_duration.sum", "sm__cycles_elapsed.avg.per_second")]
    cols = (["machine", "order", "path", "role", "kernel_name",
             "kernel_names_seen", "job", "commit", "sweep_date", "sweep_dir",
             "cutlass_mma_shape", "dmma_shape_in_sass", "dmma_flop_per_inst",
             "launches_profiled", "launches_per_call",
             "gpu__time_duration.sum_ns", "sm__cycles_elapsed.avg.per_second_hz",
             "elapsed_sm_cycles"]
            + counter_cols
            + [STALL_COL[mn] for mn in GROUP_B]
            + ["D_%s_cycles" % r for r in RESOURCES]
            + ["binding_resource", "eta"])
    rows, mismatch, n_checked = [], [], 0
    for m in sp.SWEEPS:
        for o in sp.ORDERS:
            for pth in sp.PATHS:
                d, roles = load_case_ext(m, o, pth)
                if not roles:
                    continue
                meta = metadata(d)
                for role, agg in sorted(roles.items()):
                    key = (m, str(o), pth, role)
                    base = committed.get(key)
                    if base is None:
                        mismatch.append("%s: not in committed table" % (key,))
                        continue
                    # Every shared column must reproduce the committed value.
                    n_checked += 1
                    for k in ["n", "mult", "dmma_flop"] + list(sp.METRICS):
                        mine = fmt(agg[k] if agg[k] is not None else 0.0)
                        if str(base[k]) != mine:
                            mismatch.append("%s %s: %s vs %s"
                                            % (key, k, base[k], mine))
                    # Demands on the machine the counters were taken on.
                    agg_d = dict(agg)
                    for k in sp.METRICS:
                        agg_d[k] = agg[k] if agg[k] is not None else 0.0
                    dem = sp.demands(agg_d, m)
                    eta, top = sp.eta_of(agg_d, m)
                    cyc = sp.cycles_measured(agg_d, m)
                    row = [m, o, pth, role, agg["kernel_name"],
                           agg["kernel_names_seen"], meta["job"], meta["commit"],
                           meta["date"], os.path.relpath(d, sp.ROOT), "8x8x4",
                           dmma_shape(agg["dmma_flop"]), fmt(agg["dmma_flop"]),
                           agg["n"], agg["mult"], fmt(agg["dur"]),
                           fmt(agg["freq"]), fmt(cyc)]
                    row += [fmt(agg[BASE_OF.get(mn, mn)]) for mn in counter_cols]
                    row += [fmt(agg[BASE_OF[mn]]) for mn in GROUP_B]
                    row += [fmt(dem[r]) for r in RESOURCES]
                    row += [top, fmt(eta)]
                    rows.append(row)
    if len(rows) != len(committed):
        mismatch.append("row count %d vs committed %d" % (len(rows), len(committed)))
    if mismatch:
        sys.exit("export does not reproduce analysis/subpipe_inputs.csv:\n  "
                 + "\n  ".join(mismatch[:20]))
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    print("wrote %s: %d rows, %d columns; %d rows checked against %s"
          % (a.out, len(rows), len(cols), n_checked, os.path.relpath(sp.TABLE)))
    if a.readme:
        with open(a.readme, "w") as fh:
            fh.write(README.format(n=len(rows)))
        print("wrote", a.readme)


README = """# subpipe_inputs.csv -- per-kernel Nsight Compute counters behind the sub-pipe model

One row per (machine, polynomial order, implementation path, kernel role):
{n} rows.  Generated by `analysis/subpipe_inputs_export.py` in the
scale-dg-kernel-extraction repository from the raw `ncu --csv` sweeps
(`jobs/subpipe_ncu.sh`, run through `jobs/rikyu_subpipe_*.sh`,
`jobs/tsubame_subpipe_*.sh` and `jobs/wisteria_subpipe_*.sh`).  The
counter columns the model reads are identical, value for value, to the
repository's `analysis/subpipe_inputs.csv`, which is what
`analysis/subpipe_predict.py` and `analysis/subpipe_rank_audit.py` consume;
the export script checks that and refuses to write otherwise.

## How a row is made

* Every sweep profiles one run of the benchmark at `nstep = 4`, with
  `--clock-control none` (boost clocks, not ncu's locked clock), `-s 4 -c 24`
  (skip the first four kernel launches, then profile 24), kernel filter
  `regex:tendency_|elembnd|volume_flux|flux_all|separable_lift|cutlass|gemm|Kernel`,
  `CutlassMmaShape = "8x8x4"` on every machine.
* Group A (the pipe, memory and launch counters) and group B (the issue-stall
  breakdown) are two separate `ncu` passes over different launches of the
  same run; they are merged per role.
* `role` is the kernel's canonical name across machines (`subpipe_predict.role_of`):
  our own kernels keep their symbol; cuBLAS launches, whose kernel the
  library picks per machine, collapse to `cublas_gemm` (tensor-op family) or
  `cublas_gemm_simt`.  A role that is launched more than once per tendency
  call is split into `role#1`, `role#2`, ... by occurrence, because the
  occurrences are different GEMMs (x, y, z).
* Each counter is the **median over the profiled launches of that role**
  (`launches_profiled` of them); `launches_per_call` is 1 after the split.
* Blank = the metric has no value in that sweep.  Seven counters
  (`l1tex__f_wavefronts`, the two `l1tex__t_sectors_pipe_lsu_mem_global_op_*`,
  the two `sm__inst_executed_pipe_lsu_mem_*_cmd_read`, the two
  `lts__t_sectors_lookup_*`) were added after the first GB200 order sweep
  (Slurm 92904), so its rows leave them blank; and
  `sm__inst_executed_pipe_lsu_mem_{{lg,shared}}_cmd_read` return n/a on sm_80,
  so every A100 row leaves those two blank.  None of the seven is a model
  input.

## Columns

| column | meaning |
|---|---|
| `machine` | `GB200` (RIKYU), `H100` (TSUBAME 4.0), `A100` (Wisteria Aquarius) |
| `order` | polynomial order p |
| `path` | `DqdtKernel_Type` without the `CUDAFORTRAN_` prefix |
| `role` | canonical kernel role (above) |
| `kernel_name` | the demangled kernel name most often seen for this role in this sweep |
| `kernel_names_seen` | how many distinct kernel names the role collapsed (cuBLAS > 1 possible) |
| `job`, `commit`, `sweep_date`, `sweep_dir` | scheduler job id, git commit of the profiled build, run date, local directory of the raw CSVs |
| `cutlass_mma_shape` | `CutlassMmaShape` namelist value of the run (8x8x4 throughout) |
| `dmma_shape_in_sass` | the DMMA instruction shape the kernel issues, from its name (`subpipe_predict.dmma_flop_of`): 8x8x4 for every hand-written and CUTLASS kernel and for cuBLAS on sm_80/sm_100; 16x8x8 for cuBLAS `sm90_xmma` on H100 |
| `dmma_flop_per_inst` | 2*M*N*K FLOP of that instruction |
| `launches_profiled` | number of launches the medians are taken over |
| `launches_per_call` | launches of this (split) role per tendency call |
| `gpu__time_duration.sum_ns` | kernel duration, ns (ncu `gpu__time_duration.sum`) |
| `sm__cycles_elapsed.avg.per_second_hz` | SM clock during the kernel, Hz |
| `elapsed_sm_cycles` | duration x SM clock: elapsed SM cycles of one launch |
| `sm__throughput...`, `sm__pipe_*_cycles_active...`, `sm__inst_executed*`, `smsp__inst_executed...`, `l1tex__*`, `lts__*`, `dram__*`, `sm__warps_active...`, `launch__*` | the ncu metric of that name, group A |
| `st_<reason>` | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`, group B |
| `D_shared_cycles` | model demand on the fp64+tensor shared pipe, SM cycles per launch on this machine: `(max(4, FLOP_inst / C_tc) * N_tensor + (64 / C_cc) * N_fp64) / N_SM` |
| `D_dram_cycles` | `dram__bytes.sum / (HBM bytes per SM-clock * N_SM)` |
| `D_lsu_wf_cycles` | `l1tex__data_pipe_lsu_wavefronts.sum / (1 * N_SM)` |
| `D_issue_cycles` | `sm__inst_executed.sum / (4 * N_SM)` |
| `D_fma_cycles` | `sm__inst_executed_pipe_fma.sum / (4 * N_SM)` |
| `D_alu_cycles` | `sm__inst_executed_pipe_alu.sum / (2 * N_SM)` |
| `D_l2_cycles` | `lts__t_bytes.sum / (64 * N_SM)` |
| `binding_resource` | argmax of the seven demands |
| `eta` | `elapsed_sm_cycles / max demand`: the kernel's reach of its binding roof (< 1 means the model's capacities are optimistic; > 1 means latency or a resource not in the model) |

`N_SM`, the boost clock, `C_tc` (FP64 tensor FLOP per SM-clock), `C_cc`
(FP64 CUDA-core FLOP per SM-clock) and HBM bytes per SM-clock are the
spec-sheet numbers in `subpipe_predict.MACHINES`: GB200 152 / 2.062 GHz /
127.9 / 127.9 / 25.21; A100 108 / 1.410 / 128.1 / 63.7 / 10.21; H100 132 /
1.980 / 256.0 / 128.2 / 9.17.

## Which rows the model uses

`GEMM_OZAKI1` rows are in the table for the Ozaki appendix but are not
transferred by the leave-one-GPU-out score (their ncu capture is one cuBLAS
role, not a path total).  The sub-pipe report's section 5AB.2 says why.
"""


if __name__ == "__main__":
    main()
