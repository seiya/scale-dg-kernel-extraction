#!/usr/bin/env python3
"""Export the closure of the shared sub-pipe balance, kernel by kernel.

The manuscript prices every instruction that reaches the fp64+tensor shared
pipe (Eq. subpipe, reports/subpipe_transfer_model.md section 2 and
analysis/subpipe_predict.demands):

    S_pred = c_tc * N_DMMA + c_cc * N_fp64
    c_tc   = max(4, F_DMMA / C_tc,m)         c_cc = 64 / C_cc,m

and the reviewer asked to see, in one table rather than in prose, on how
wide a kernel set that balance closes against the counter that measures the
pipe.  This script writes that table from the published per-kernel counter
table (subpipe_inputs_export.py, one row per machine/order/path/role; 666
rows) and nothing else, so the manuscript side can re-derive every number
from the distributed CSV alone.  Definitions, per row:

    elapsed_sm_cycles       = gpu__time_duration * sm__cycles_elapsed.avg.per_second
                              (the SM clock the launch recorded, not the boost clock)
    shared_cycles_measured  = sm__pipe_shared_cycles_active.pct / 100
                              * elapsed_sm_cycles * N_SM
    shared_cycles_predicted = c_tc * N_DMMA + c_cc * N_fp64
    ratio                   = measured / predicted
    cycles_per_dmma_measured     = shared_cycles_measured / N_DMMA
    cycles_per_dmma_net_measured = (shared_cycles_measured - c_cc * N_fp64) / N_DMMA
    cycles_per_fp64_measured     = shared_cycles_measured / N_fp64   (N_DMMA = 0 only)
    ratio_nonmma_tensor_at_half  = shared_cycles_measured
                                   / (c_cc * N_fp64 + 0.5 * N_DMMA)   (issues_dmma = 0 only)

with N_DMMA = sm__inst_executed_pipe_tensor.sum and N_fp64 =
sm__inst_executed_pipe_fp64.sum, both the per-role medians of the counter
table.  Every row of the counter table is written; `included` marks the rows
the closure statistic is taken over: shared-pipe utilisation above 5 %, a
launch of at least N_SM blocks (the measured side multiplies a per-active-SM
percentage by N_SM, which is wrong for the 1-4 block cuBLAS launches of the
low-order z GEMM; those read 70x-370x), the Ozaki path left out for the
reason in section 5AB.2, and at least one priced instruction.  Filtering
happens in the CSV, not outside it.

`issues_dmma` says whether the kernel's source issues an mma at all, from
its identity (cuBLAS tensor-op family, CUTLASS, the FUSED_TC kernels), not
from the counter: on sm_80 and sm_90 sm__inst_executed_pipe_tensor is
non-zero in kernels that contain no mma (separable_lift, volume_flux,
elembnd_flux, the cuBLAS SIMT kernels), at a fixed fraction of N_fp64 (1/13,
1/7, 2/13, 1/3), and those instructions occupy the pipe for about 0.5 cycle,
not 4.  Eq. subpipe prices them at 4, so it over-predicts those kernels
(ratio 0.50-0.82); none of them is shared-pipe bound, and they are kept in
the table under `instruction_class` = fp64_only so the miss is visible.  On
sm_100 the counter is zero in every such kernel.

    python3 analysis/subpipe_closure_export.py \
        --inputs  ../scale-dg-paper/figs/data/subpipe_inputs.csv \
        --out     ../scale-dg-paper/figs/data/subpipe_closure.csv \
        --summary ../scale-dg-paper/figs/data/subpipe_closure_summary.csv

The summary has one row per (machine, instruction class, order band) plus
"all" rows, over the included rows only: kernel count, and median / min /
max of ratio, cycles_per_dmma_measured and cycles_per_dmma_net_measured.
Section 5AK of reports/subpipe_transfer_model.md records what this export
found when it was first run.
"""
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subpipe_predict as sp                                   # noqa: E402

UTIL_MIN_PCT = 5.0          # shared-pipe utilisation floor for `included`
EXCLUDED_PATHS = ("GEMM_OZAKI1",)

SHARED_PCT = "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active"
N_TENSOR = "sm__inst_executed_pipe_tensor.sum"
N_FP64 = "sm__inst_executed_pipe_fp64.sum"

COLS = ["machine", "order", "path", "role", "issues_dmma", "instruction_class",
        "dmma_shape_in_sass", "dmma_flop_per_inst", "N_SM", "grid_blocks",
        "elapsed_sm_cycles", "shared_util_pct",
        "N_DMMA", "N_fp64", "c_tc_predicted", "c_cc_predicted",
        "shared_cycles_measured", "shared_cycles_predicted", "ratio",
        "cycles_per_dmma_measured", "cycles_per_dmma_net_measured",
        "cycles_per_fp64_measured", "ratio_nonmma_tensor_at_half", "job",
        "included"]

SUMMARY_COLS = ["machine", "instruction_class", "order_band", "n_kernels",
                "ratio_median", "ratio_min", "ratio_max",
                "cycles_per_dmma_median", "cycles_per_dmma_min",
                "cycles_per_dmma_max",
                "cycles_per_dmma_net_median", "cycles_per_dmma_net_min",
                "cycles_per_dmma_net_max"]


def fmt(v, prec=6):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    return "%.*g" % (prec, v)


def issues_dmma(path, role):
    """Does this kernel's source issue an mma?  By identity, not by counter."""
    if path in EXCLUDED_PATHS:
        return False                      # Ozaki: int8 IMMA, not DMMA
    if role.startswith("cublas_gemm") and not role.startswith("cublas_gemm_simt"):
        return True
    if role.startswith("cutlass"):
        return True
    return path == "FUSED_TC" and role.startswith("tendency_")


def closure_row(r):
    m = sp.MACHINES[r["machine"]]
    n_sm = m["sm"]
    n_dmma = float(r[N_TENSOR])
    n_fp64 = float(r[N_FP64])
    flop = float(r["dmma_flop_per_inst"])
    util = float(r[SHARED_PCT])
    cyc = float(r["elapsed_sm_cycles"])
    c_tc = max(4.0, flop / m["fp64_tc"])
    c_cc = 64.0 / m["fp64_cc"]
    meas = util / 100.0 * cyc * n_sm
    pred = c_tc * n_dmma + c_cc * n_fp64
    dmma = issues_dmma(r["path"], r["role"])
    grid = int(float(r["launch__grid_size"]))
    if dmma:
        klass = "DMMA." + r["dmma_shape_in_sass"]
    elif n_dmma > 0 and r["path"] in EXCLUDED_PATHS:
        klass = "tensor_other"
    elif n_fp64 > 0 or n_dmma > 0:
        klass = "fp64_only"
    else:
        klass = "none"
    ratio = meas / pred if pred > 0 else None
    cpd = meas / n_dmma if n_dmma > 0 else None
    cpd_net = (meas - c_cc * n_fp64) / n_dmma if n_dmma > 0 else None
    cpf = meas / n_fp64 if (n_dmma == 0 and n_fp64 > 0) else None
    # For a kernel that issues no mma, the same balance with the counted
    # non-mma "tensor" instructions at the 0.5 cycle they measure (docstring).
    alt = None
    if not dmma and n_dmma > 0 and n_fp64 > 0:
        alt = meas / (c_cc * n_fp64 + 0.5 * n_dmma)
    included = int(util > UTIL_MIN_PCT
                   and grid >= n_sm
                   and r["path"] not in EXCLUDED_PATHS
                   and pred > 0)
    return {
        "machine": r["machine"], "order": int(r["order"]), "path": r["path"],
        "role": r["role"], "issues_dmma": int(dmma), "instruction_class": klass,
        "dmma_shape_in_sass": r["dmma_shape_in_sass"],
        "dmma_flop_per_inst": fmt(flop), "N_SM": n_sm, "grid_blocks": grid,
        "elapsed_sm_cycles": r["elapsed_sm_cycles"], "shared_util_pct": r[SHARED_PCT],
        "N_DMMA": r[N_TENSOR], "N_fp64": r[N_FP64],
        "c_tc_predicted": fmt(c_tc), "c_cc_predicted": fmt(c_cc),
        "shared_cycles_measured": fmt(meas), "shared_cycles_predicted": fmt(pred),
        "ratio": fmt(ratio), "cycles_per_dmma_measured": fmt(cpd),
        "cycles_per_dmma_net_measured": fmt(cpd_net),
        "cycles_per_fp64_measured": fmt(cpf),
        "ratio_nonmma_tensor_at_half": fmt(alt),
        "job": r["job"], "included": included,
        # unformatted, for the summary; the per-DMMA costs only where the
        # kernel issues one
        "_ratio": ratio, "_cpd": cpd if dmma else None,
        "_cpd_net": cpd_net if dmma else None,
    }


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ["", "", ""]
    return [fmt(statistics.median(vals), 4), fmt(min(vals), 4), fmt(max(vals), 4)]


def summarize(rows):
    inc = [r for r in rows if r["included"]]
    machines = [m for m in sp.MACHINES if any(r["machine"] == m for r in inc)]
    classes = sorted({r["instruction_class"] for r in inc},
                     key=lambda k: (not k.startswith("DMMA"), k))
    bands = list(sp.ORDER_BANDS) + ["all"]
    out = []
    for m in machines:
        for k in classes + ["all"]:
            for b in bands:
                sel = [r for r in inc if r["machine"] == m
                       and (k == "all" or r["instruction_class"] == k)
                       and (b == "all" or sp.order_band(r["order"]) == b)]
                if not sel:
                    continue
                out.append([m, k, b, len(sel)]
                           + stats([r["_ratio"] for r in sel])
                           + stats([r["_cpd"] for r in sel])
                           + stats([r["_cpd_net"] for r in sel]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="the published counter table (subpipe_inputs_export.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()

    with open(a.inputs) as fh:
        src = list(csv.DictReader(fh))
    # Same row set as the model's own table, in its order.
    with open(sp.TABLE) as fh:
        keys = [(r["machine"], r["order"], r["path"], r["role"])
                for r in csv.DictReader(fh)]
    by_key = {(r["machine"], r["order"], r["path"], r["role"]): r for r in src}
    missing = [k for k in keys if k not in by_key]
    if missing or len(src) != len(keys):
        sys.exit("%s does not have the %d rows of %s (missing %d, extra %d)"
                 % (a.inputs, len(keys), os.path.relpath(sp.TABLE),
                    len(missing), len(src) - len(keys) + len(missing)))
    rows = [closure_row(by_key[k]) for k in keys]

    # The predicted cycles must agree with the model's own D_shared_cycles
    # column of the same table (6 significant digits), or the two would be
    # two different equations.
    for k, r in zip(keys, rows):
        d = float(by_key[k]["D_shared_cycles"]) * sp.MACHINES[r["machine"]]["sm"]
        p = float(r["shared_cycles_predicted"])
        if d > 0 and abs(p / d - 1.0) > 2e-5:
            sys.exit("%s: shared_cycles_predicted %g vs D_shared_cycles*N_SM %g"
                     % (k, p, d))

    # On sm_100 the tensor counter is zero exactly in the kernels that issue
    # no mma, so the identity rule above can be checked against it there.
    for r in rows:
        if r["machine"] == "GB200" and r["path"] not in EXCLUDED_PATHS:
            if bool(r["issues_dmma"]) != (float(r["N_DMMA"]) > 0):
                sys.exit("GB200 %s/%s/%s: issues_dmma=%d but N_DMMA=%s"
                         % (r["order"], r["path"], r["role"], r["issues_dmma"],
                            r["N_DMMA"]))

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLS)
        for r in rows:
            w.writerow([r[c] for c in COLS])
    summ = summarize(rows)
    with open(a.summary, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(SUMMARY_COLS)
        w.writerows(summ)
    n_inc = sum(r["included"] for r in rows)
    print("wrote %s: %d rows, %d included (util > %g%%, grid >= N_SM, paths %s "
          "excluded)" % (a.out, len(rows), n_inc, UTIL_MIN_PCT,
                         "/".join(EXCLUDED_PATHS)))
    print("wrote %s: %d rows" % (a.summary, len(summ)))


if __name__ == "__main__":
    main()
