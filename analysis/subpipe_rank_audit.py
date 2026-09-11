#!/usr/bin/env python3
"""Score the sub-pipe model's *ranking* claim, on both axes, two ways.

reports/subpipe_transfer_model.md sections 5W / 5Y.3 / 5Z.10 published a
threshold rule: "if the predicted difference exceeds the model's own error by
1.1x, the rank is right".  The scoring behind it used

    S/N = |r_meas - 1| / |r_pred / r_meas - 1|          (posterior)

whose numerator and denominator both contain the target machine's measured
ratio, so it cannot be evaluated before measuring the target -- and, worse, a
sign flip at S/N >= t is arithmetically impossible unless r_meas > t, which is
why that side scored 100%.  Section 5AA re-scores the same cells with the
rule as it is actually stated in the report,

    S/N = |r_pred - 1| / (leave-one-out median error of that ratio's family)
                                                        (prior)

which uses no target measurement at all.  This script computes both.

Machine axis: 3 leave-one-GPU-out directions x 7 ratios x all orders, from the
counters in analysis/subpipe_inputs.csv (or the local ncu sweeps).
Balance axis: 2 machines x 3 orders x 7 ratios x 5 MPS fractions.  Predictions
come from the same predictor with N_SM set to the measured SM count and DRAM
bytes per SM-clock scaled by SM_full / N_SM (MPS fences off SMs, not HBM);
measured times are the distilled medians in analysis/subpipe_balance_times.csv.

  python3 analysis/subpipe_rank_audit.py [--axis machine|balance|both]
"""
import argparse
import collections
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subpipe_predict as sp                                   # noqa: E402

BAL_TIMES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "subpipe_balance_times.csv")
LOGO = [(["A100", "H100"], "GB200"), (["GB200", "H100"], "A100"),
        (["GB200", "A100"], "H100")]


def cell(axis, tag, ratio, rp, rm):
    return dict(axis=axis, tag=tag, ratio=ratio, rp=rp, rm=rm,
                err=abs(rp / rm - 1.0) * 100.0, hit=(rp > 1) == (rm > 1))


def machine_axis():
    cells = []
    for srcs, tgt in LOGO:
        got = {}
        for p in sp.ORDERS:
            for path in sp.PATHS:
                pr, me, _ = sp.predict_case(srcs, tgt, p, path)
                if pr is None or not me:
                    continue
                got.setdefault(p, {})[path] = (pr, me)
        for num, den in sp.RATIOS:
            for p, r in sorted(got.items()):
                if num in r and den in r:
                    cells.append(cell("machine", "%s p=%d" % (tgt, p),
                                      num + " / " + den,
                                      r[num][0] / r[den][0],
                                      r[num][1] / r[den][1]))
    return cells


def balance_axis():
    meas, sm_of = {}, {}
    with open(BAL_TIMES) as fh:
        for r in csv.DictReader(fh):
            key = (r["machine"], int(r["order"]), int(r["pct"]))
            meas.setdefault(key, {})[r["path"]] = float(r["device_s_median"])
            sm_of[key] = int(r["sm"])
    pred = {}
    for (mach, order, pct), sm in sorted(sm_of.items()):
        key = "%s@%dpct" % (mach, pct)
        full = sp.MACHINES[mach]["sm"]
        spec = dict(sp.MACHINES[mach])
        spec["sm"] = sm
        spec["dram_B"] = sp.MACHINES[mach]["dram_B"] * full / sm
        sp.MACHINES[key] = spec
        sp.SWEEPS[key] = []                     # no sweep of its own: predict only
        for path in meas[(mach, order, pct)]:
            p, _, _ = sp.predict_case([mach], key, order, path)
            if p:
                pred.setdefault((mach, order, pct), {})[path] = p
    cells = []
    for k in sorted(meas):
        if k[2] == 100:                          # the calibration point itself
            continue
        for num, den in sp.RATIOS:
            m, q = meas[k], pred.get(k, {})
            if num in m and den in m and num in q and den in q:
                cells.append(cell("balance", "%s p=%d %d%%" % (k[0], k[1], k[2]),
                                  num + " / " + den,
                                  q[num] / q[den], m[num] / m[den]))
    return cells


def score(cells, label):
    fam = collections.defaultdict(list)
    for c in cells:
        fam[c["ratio"]].append(c["err"])
    for c in cells:
        rest = list(fam[c["ratio"]])
        rest.remove(c["err"])
        c["fam_loo"] = st.median(rest) if rest else c["err"]
        c["sn_post"] = (abs(c["rm"] - 1) * 100 / c["err"]) if c["err"] else 9e9
        c["sn_prior"] = abs(c["rp"] - 1) * 100 / c["fam_loo"]
    print("\n=== %s: %d cells, %d ranked right (%.0f%%)"
          % (label, len(cells), sum(c["hit"] for c in cells),
             100 * sum(c["hit"] for c in cells) / len(cells)))
    for key, name in (("sn_post", "posterior S/N (published: uses r_meas)"),
                      ("sn_prior", "prior S/N (|r_pred-1| / family error)")):
        print("  %s" % name)
        for t in (1.0, 1.1, 1.5, 2.0):
            hi = [c for c in cells if c[key] >= t]
            lo = [c for c in cells if c[key] < t]
            print("    t=%.1f  >=t %3d/%3d = %5.1f%%   <t %3d/%3d = %5.1f%%"
                  % (t, sum(c["hit"] for c in hi), len(hi),
                     100 * sum(c["hit"] for c in hi) / max(1, len(hi)),
                     sum(c["hit"] for c in lo), len(lo),
                     100 * sum(c["hit"] for c in lo) / max(1, len(lo))))
    t = 1.1
    hi = [c for c in cells if c["sn_post"] >= t]
    safe = [c for c in hi if c["rm"] < t]
    rest = [c for c in hi if c["rm"] >= t]
    req = [abs(c["rm"] - 1) / c["rm"] * 100 for c in rest]
    print("  why the posterior side is 100%%: of the %d cells at S/N >= 1.1, "
          "%d cannot flip at all (r_meas < 1.1);" % (len(hi), len(safe)))
    if rest:
        print("    the other %d would need a relative error of %.1f%% (median, "
              "min %.1f%%) to flip, against an actual %.1f%% (median)"
              % (len(rest), st.median(req), min(req),
                 st.median([c["err"] for c in rest])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="both",
                    choices=["machine", "balance", "both"])
    ap.add_argument("--from-table", action="store_true",
                    help="read counters from analysis/subpipe_inputs.csv")
    a = ap.parse_args()
    sp.USE_TABLE[0] = a.from_table
    cells = []
    if a.axis in ("machine", "both"):
        m = machine_axis()
        score(m, "machine axis (leave-one-GPU-out)")
        cells += m
    if a.axis in ("balance", "both"):
        b = balance_axis()
        score(b, "balance axis (MPS SM fraction)")
        cells += b
    if len(cells) > len(m if a.axis == "machine" else []) and a.axis == "both":
        score(cells, "both axes")


if __name__ == "__main__":
    main()
