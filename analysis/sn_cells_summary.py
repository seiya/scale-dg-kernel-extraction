#!/usr/bin/env python3
"""The numbers the paper's S/N text quotes, re-derived from the distributed
CSVs alone (no predictor run): figs/data/sn_cells.csv,
figs/data/sn_cells_all_pairs.csv and figs/data/logo_path_totals.csv, as
written by

  python3 analysis/subpipe_rank_audit.py --axis both --from-table \\
      --famerr isolated --dump-cells sn_cells.csv \\
      --dump-all-pairs sn_cells_all_pairs.csv
  python3 analysis/subpipe_predict.py --logo --from-table --dump-logo \\
      logo_path_totals.csv

Scoring convention (report section 5AE.0): a machine-axis cell is scored on
its ledger-consistent subset with the `_led` column, a balance-axis cell with
the plain column.  The paper scores every ranking in the log-symmetric form
(W1, report section 5AJ.6): S/N = |ln r_pred| / median |ln(r_pred / r_meas)|,
i.e. the `sn_log` / `sn_led_log` columns; the linear `sn` / `sn_led` columns
are kept in the CSV as the record of the earlier form and are not quoted by
the paper.  Four blocks (report sections 5AJ.1-5AJ.3 and 5AJ.6):

  1. threshold table of the linear S/N (`sn`) against the log-symmetric one
     (`sn_log`), per axis, with the cells whose admission differs;
  2. the admitted cells split by run-to-run width (W2): with a width, with
     the rank resolved by the width, without a width;
  3. the fastest-path choice in the leave-one-GPU-out cells: predicted vs
     measured fastest, S/N against the nearest predicted competitor (the
     paper's Table "fastest-path") and, from the all-pairs file (W3), the
     minimum S/N over ALL competitors;
  4. the rest of what the paper's S/N text quotes: the forced sign below the
     threshold, the split by the two weak families, the misses above the
     threshold, the a posteriori form, and the per-ratio tables at p >= 63
     and at p <= 15 (true difference and error as median |ln r|).

  python3 analysis/sn_cells_summary.py DATA_DIR [--col sn_log|sn]
"""
import argparse
import collections
import csv
import math
import os
import statistics as st

THRESH = (1.0, 1.1, 1.5, 2.0)


def load(fn):
    with open(fn, newline="") as fh:
        return list(csv.DictReader(fh))


def scored(rows, col):
    """(cell, S/N, hit) of every cell the convention scores, with the column
    the convention uses for it."""
    out = []
    for r in rows:
        if r["axis"] == "machine":
            if r["ledger_ok"] != "1":
                continue
            v = r[col.replace("sn", "sn_led", 1)]
        else:
            v = r[col]
        out.append((r, float(v), r["hit"] == "1"))
    return out


def tag(r):
    s = "%s %s p=%s %s/%s" % (r["axis"], r["grp"], r["order"], r["num"], r["den"])
    return s + (" %s%%" % r["pct"] if r.get("pct") else "")


def block1(cells):
    print("== 1. linear S/N (sn) against the log-symmetric S/N (sn_log, the "
          "paper's form)")
    for axis in ("machine", "balance"):
        print("  %s axis (%s)" % (axis, "sn_led on ledger_ok=1 cells"
                                  if axis == "machine" else "sn"))
        a = [x for x in scored(cells, "sn") if x[0]["axis"] == axis]
        b = [x for x in scored(cells, "sn_log") if x[0]["axis"] == axis]
        for t in THRESH:
            ha = [x for x in a if x[1] >= t]
            hb = [x for x in b if x[1] >= t]
            print("    t=%.1f  sn %3d/%3d = %5.1f%%   sn_log %3d/%3d = %5.1f%%"
                  % (t, sum(h for _, _, h in ha), len(ha),
                     100.0 * sum(h for _, _, h in ha) / max(1, len(ha)),
                     sum(h for _, _, h in hb), len(hb),
                     100.0 * sum(h for _, _, h in hb) / max(1, len(hb))))
            ia = {id(x[0]) for x in ha}
            ib = {id(x[0]) for x in hb}
            for r, v, h in a:
                if (id(r) in ia) != (id(r) in ib):
                    vb = next(y[1] for y in b if y[0] is r)
                    print("          %s: sn %.4f sn_log %.4f -> %s, %s"
                          % (tag(r), v, vb,
                             "enters" if id(r) in ib else "leaves",
                             "hit" if h else "miss"))


def block2(cells, col, t=1.0):
    print("\n== 2. admitted cells (%s >= %.1f) by run-to-run width" % (col, t))
    for axis in ("machine", "balance"):
        adm = [x for x in scored(cells, col) if x[0]["axis"] == axis and x[1] >= t]
        w = [x for x in adm if x[0]["width"] != ""]
        res = [x for x in w if x[0]["resolved"] == "1"]
        unres = [x for x in w if x[0]["resolved"] == "0"]
        nw = [x for x in adm if x[0]["width"] == ""]
        f = lambda xs: "%d/%d" % (sum(h for _, _, h in xs), len(xs))
        print("  %s: admitted %s; with width %s; resolved=1 %s; resolved=0 %s; "
              "no width %s" % (axis, f(adm), f(w), f(res), f(unres), f(nw)))
        for r, v, h in unres:
            print("      resolved=0: %s  |r_meas-1| %.2f%% width %s%% %s"
                  % (tag(r), abs(float(r["meas"]) - 1) * 100, r["width"],
                     "hit" if h else "miss"))
        by = collections.Counter("%s p=%s" % (r["grp"], r["order"]) for r, _, _ in nw)
        if by:
            print("      no width: " + ", ".join("%s x%d" % kv for kv in sorted(by.items())))


def block3(logo, pairs, col, led):
    """`led` False reads the plain column, which is what the paper's Table
    "fastest-path" printed (its S/N 2.30 at GB200 p=127 is `sn`, not
    `sn_led` 2.09); True reads the `_led` column of the section-5AE.0
    convention.  Both are on ledger-consistent cells only."""
    lcol = col.replace("sn", "sn_led", 1) if led else col
    print("\n== 3. fastest path per (target, order), ledger-consistent paths; "
          "S/N column %s" % lcol)
    sn = {}
    for r in pairs:
        if r["axis"] == "machine" and r["ledger_ok"] == "1":
            sn[(r["grp"], int(r["order"]), r["num"], r["den"])] = float(r[lcol])
    cells = collections.defaultdict(dict)
    for r in logo:
        if r["ledger_ok"] == "1":
            cells[(r["tgt"], int(r["order"]))][r["path"]] = (float(r["pred"]),
                                                             float(r["meas"]))
    n_right = n_near = n_near_right = n_all = n_all_right = n_unscored = 0
    print("  %-6s %4s %2s %-11s %-11s %5s  %-11s %7s  %-11s %7s  claim(near/all)"
          % ("target", "p", "n", "pred-fast", "meas-fast", "loss%", "nearest",
             "S/N", "min-comp", "minS/N"))
    for (tgt, p), d in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(d) < 2:
            continue
        byp = sorted(d, key=lambda k: d[k][0])
        pf, near = byp[0], byp[1]
        mf = min(d, key=lambda k: d[k][1])
        loss = (d[pf][1] / d[mf][1] - 1) * 100
        right = pf == mf
        n_right += right
        def sn_of(a, b):
            return sn.get((tgt, p, a, b), sn.get((tgt, p, b, a)))
        s_near = sn_of(pf, near)
        comp = [(sn_of(pf, c), c) for c in d if c != pf]
        if any(v is None for v, _ in comp):
            n_unscored += 1
            s_min, c_min = None, "unscored"
        else:
            s_min, c_min = min(comp)
        claim_near = s_near is not None and s_near >= 1.0
        claim_all = s_min is not None and s_min >= 1.0
        n_near += claim_near
        n_near_right += claim_near and right
        n_all += claim_all
        n_all_right += claim_all and right
        print("  %-6s %4d %2d %-11s %-11s %5.1f  %-11s %7s  %-11s %7s  %s/%s"
              % (tgt, p, len(d), pf, mf, loss, near,
                 "%.2f" % s_near if s_near is not None else "n/a", c_min,
                 "%.2f" % s_min if s_min is not None else "n/a",
                 "yes" if claim_near else "no", "yes" if claim_all else "no"))
    print("  cells %d; predicted fastest = measured fastest %d; nearest-competitor "
          "S/N>=1 claims %d, right %d; all-competitor S/N>=1 claims %d, right %d; "
          "cells with an unscored pair %d"
          % (len([1 for d in cells.values() if len(d) >= 2]), n_right, n_near,
             n_near_right, n_all, n_all_right, n_unscored))


WEAK = {("GEMM_CUTE", "GEMM"), ("GEMM_CUTE", "GEMM_FUSED")}


def block4(cells, col, t=1.0):
    """The remaining numbers of the paper's S/N text (report section 5AJ.6)."""
    use_log = col == "sn_log"
    print("\n== 4. the rest of the S/N text, column %s, threshold %.1f" % (col, t))
    f = lambda xs: "%d/%d" % (sum(h for _, _, h in xs), len(xs))
    for axis in ("machine", "balance"):
        sc = [x for x in scored(cells, col) if x[0]["axis"] == axis]
        above = [x for x in sc if x[1] >= t]
        below = [x for x in sc if x[1] < t]
        weak = lambda x: (x[0]["num"], x[0]["den"]) in WEAK
        miss = [x for x in above if not x[2]]
        print("  %s: above %s, forced sign below %s = %.1f%%; without the two "
              "weak families above %s of %d cells, on them alone %s of %d; "
              "misses above %d, in the two families %d"
              % (axis, f(above), f(below),
                 100.0 * sum(h for _, _, h in below) / max(1, len(below)),
                 f([x for x in above if not weak(x)]),
                 len([x for x in sc if not weak(x)]),
                 f([x for x in above if weak(x)]),
                 len([x for x in sc if weak(x)]),
                 len(miss), len([x for x in miss if weak(x)])))
    # a posteriori form: the measured distance from 1 against the error the
    # cell itself realised.  Clearing it implies the correct sign, so the
    # hit count is a check, not a result.
    sc = scored(cells, col)
    post = []
    for r, _, h in sc:
        rp, rm = float(r["pred"]), float(r["meas"])
        d = (abs(math.log(rm)) >= abs(math.log(rp / rm)) if use_log
             else abs(rm - 1) >= abs(rp - rm))
        if d:
            post.append(h)
    print("  a posteriori form (%s): %d of %d cells clear it, %d with the "
          "correct sign"
          % ("|ln r_meas| >= |ln(r_pred/r_meas)|" if use_log
             else "|r_meas-1| >= |r_pred-r_meas|", len(post), len(sc),
             sum(post)))
    # the partial-sum cells (machine axis, ledger_ok=0): sign only
    ps = [r for r in cells if r["axis"] == "machine" and r["ledger_ok"] != "1"]
    ab = [r for r in ps if float(r[col]) >= t]
    print("  partial-sum cells (not scored) %d; %s >= %.1f: %d, with the sign "
          "of the partial-sum measurement %d"
          % (len(ps), col, t, len(ab), sum(r["hit"] == "1" for r in ab)))
    # per-ratio tables: signs at p >= 63 (the paper's Table "seven-ratios")
    # and the criterion at p <= 15 (Table "low-order-rankings")
    dist = ((lambda x: abs(math.log(x))) if use_log else (lambda x: abs(x - 1)))
    err = ((lambda a, b: abs(math.log(a / b))) if use_log
           else (lambda a, b: abs(a / b - 1)))
    led = [x for x in sc if x[0]["axis"] == "machine"]
    for name, sel in (("p >= 63, sign", lambda r: int(r["order"]) >= 63),
                      ("p <= 15, criterion", lambda r: int(r["order"]) <= 15)):
        by = collections.defaultdict(list)
        for x in led:
            if sel(x[0]):
                by[(x[0]["num"], x[0]["den"])].append(x)
        print("  machine axis, %s: %d cells" % (name, sum(map(len, by.values()))))
        tot = [0, 0, 0, 0]
        for k, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            rp = [float(r["pred"]) for r, _, _ in xs]
            rm = [float(r["meas"]) for r, _, _ in xs]
            acc = [x for x in xs if x[1] >= t]
            bel = [x for x in xs if x[1] < t]
            tot[0] += sum(h for _, _, h in acc); tot[1] += len(acc)
            tot[2] += sum(h for _, _, h in bel); tot[3] += len(bel)
            print("    %-22s n %2d sign %5s  accepted %5s  below %5s  true "
                  "diff %5.1f%%  error %5.1f%%"
                  % ("/".join(k), len(xs), f(xs), f(acc), f(bel),
                     100 * st.median(dist(x) for x in rm),
                     100 * st.median(err(a, b) for a, b in zip(rp, rm))))
        print("    total: accepted %d/%d, below %d/%d" % tuple(tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--col", default="sn_log", choices=["sn_log", "sn"],
                    help="S/N column for blocks 2-4 (block 1 shows both); "
                         "the paper's form is sn_log")
    a = ap.parse_args()
    cells = load(os.path.join(a.data_dir, "sn_cells.csv"))
    pairs = load(os.path.join(a.data_dir, "sn_cells_all_pairs.csv"))
    logo = load(os.path.join(a.data_dir, "logo_path_totals.csv"))
    block1(cells)
    block2(cells, a.col)
    block3(logo, pairs, a.col, led=False)
    block3(logo, pairs, a.col, led=True)
    block4(cells, a.col)


if __name__ == "__main__":
    main()
