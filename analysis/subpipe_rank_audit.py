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

Two further defects of the scoring, fixed 2026-09-12 (V2 / V3):

V2  The ratio cells were scored without looking at `ledger_ok`, the fourth
    return value of `predict_case`.  A cell whose calibration and target kernel
    ledgers disagree has pred and meas summed over the INTERSECTION of the two
    ledgers, so its ratio is a ratio of partial sums and says nothing about
    which path is faster overall.  18 of the 151 machine-axis cells have at
    least one such path in them (all at p = 7 and p = 15).  They are now scored
    in a separate bucket, and `ledger_ok` is a column of --dump-cells.

V3  The family error that the prior S/N divides by was built from the OTHER
    hold-out directions -- but with three machines and two-machine calibration,
    every other direction uses the target machine as a calibration source.  So
    "no information about the target machine enters the decision" still did not
    hold.  The only transfers in which the target never appears at all are the
    SINGLE-machine transfers between the other two, and --famerr isolated uses
    exactly those.  That is a different calibration configuration, which is a
    constraint, not a feature: report_axis() prints how much bigger or smaller
    the isolated pool's errors are.

Three additions of 2026-09-19 (W1 / W2 / W3), all as NEW columns or a NEW file
so that the columns and rows of the distributed sn_cells.csv stay byte-equal:

W1  A log-symmetric S/N beside the published one.  |r_pred - 1| and the
    family's |r_pred / r_meas - 1| both change value when a ratio A/B is
    written as B/A, so the published rule can admit a cell in one spelling
    and reject it in the other.  famerr_log = median |ln(r_pred / r_meas)|
    over the same pool as famerr, sn_log = |ln r_pred| / famerr_log; both are
    invariant under A/B <-> B/A.  Columns famerr_log / sn_log, and
    famerr_led_log / sn_led_log on the ledger-consistent cells.

W2  The run-to-run width of every cell's measured ratio, from the repeat jobs
    (REPEATS below; report sections 5Y.2 and 5AE.2-5AE.5).  width is
    (max - min) / min of the ratio over the rounds of ONE job, in percent,
    with each round's path total summed over the SAME kernel ledger the cell
    is scored on (the intersection of the calibration and target ledgers, V2),
    so that the width and the scored r_meas are the same quantity.  Empty
    when no repeat job covers both paths of the ratio at that order.
    resolved = 1 when |r_meas - 1| exceeds the width, i.e. when the
    measurement itself settles the rank; 0 when it does not; empty with the
    width.  width_job names the job the width came from.

W3  --dump-all-pairs writes the same schema for all 15 pairs of the six
    transferred paths (ALL_PAIRS), not only the 7 one-axis ratios of
    subpipe_predict.RATIOS, so the predicted-fastest path of a (target,
    order) can be scored against EVERY competitor.  RATIOS itself is not
    changed; the 7 published ratios' rows in that file are identical to
    sn_cells.csv, since a ratio's family error only ever pools that ratio.

Machine axis: 3 leave-one-GPU-out directions x 7 ratios x all orders, from the
counters in analysis/subpipe_inputs.csv (or the local ncu sweeps).
Balance axis: 2 machines x 3 orders x 7 ratios x 5 MPS fractions.  Predictions
come from the same predictor with N_SM set to the measured SM count and DRAM
bytes per SM-clock scaled by SM_full / N_SM (MPS fences off SMs, not HBM);
measured times are the distilled medians in analysis/subpipe_balance_times.csv.

  python3 analysis/subpipe_rank_audit.py [--axis machine|balance|both]
                                        [--famerr loo-group|isolated]
                                        [--dump-cells F] [--dump-all-pairs F]
"""
import argparse
import collections
import csv
import glob
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subpipe_predict as sp                                   # noqa: E402

BAL_TIMES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "subpipe_balance_times.csv")
LOGO = [(["A100", "H100"], "GB200"), (["GB200", "H100"], "A100"),
        (["GB200", "A100"], "H100")]

# V3: the only transfers in which a given machine appears NOWHERE -- not as the
# target and not as a calibration source.  With three machines these are the
# single-machine transfers between the other two.  There is no two-machine
# calibration that excludes the target, which is why --famerr isolated has to
# change the calibration configuration and not just the pool.
SINGLE_POOL = {
    "GB200": [(["A100"], "H100"), (["H100"], "A100")],
    "A100":  [(["GB200"], "H100"), (["H100"], "GB200")],
    "H100":  [(["GB200"], "A100"), (["A100"], "GB200")],
}
FAMERR = ["loo-group"]

# W3.  Every pair of the six transferred paths, oriented like the 7 published
# ratios: the denominator is the path that comes first in LOGO_PATHS
# (FUSED_TC, GEMM_FUSED, GEMM, FUSED, FUSED_DFMA, GEMM_CUTE), which puts
# FUSED_TC under everything and GEMM_CUTE over everything, exactly as RATIOS
# has them.  The 7 published pairs come first, in RATIOS order, so that the
# rows of --dump-all-pairs restricted to them are --dump-cells row for row.
ALL_PAIRS = list(sp.RATIOS) + [
    (sp.LOGO_PATHS[j], sp.LOGO_PATHS[i])
    for i in range(len(sp.LOGO_PATHS))
    for j in range(i + 1, len(sp.LOGO_PATHS))
    if (sp.LOGO_PATHS[j], sp.LOGO_PATHS[i]) not in sp.RATIOS]
assert len(ALL_PAIRS) == 15 and len(set(ALL_PAIRS)) == 15

# W2.  The repeat jobs, per machine, in the order they are searched: the first
# job that covers BOTH paths of a ratio at that order, with every kernel role
# the cell is scored on, in at least two rounds, supplies the width.  Each
# entry is one job, given as a list of directories because SGE 8650257 hit
# its TIME_BUDGET after two rounds and 8650967 / 8652177 added rounds three
# and four (section 5AE.5); they are one logical job and are pooled, and the
# round key is (directory, round) so that their r01s are not merged.  Jobs are
# never pooled otherwise: the width is within one job, as sections 5Y.2 and
# 5AE define it (between jobs the same ratio moves up to 1.14% on GB200 and
# the A100 changes device, section 5AE.6 / 5Y.2 note).  Directories that are
# not present (the raw ncu output stays local) simply supply nothing.
REPEATS = {
    "GB200": [["subpipe_repeat_106648"],            # p=7/15/31, 6 paths, 3 rounds
              ["subpipe_repeat_95805"],             # p=63/511/575, 4 paths, 3
              ["subpipe_repeat_106952"]],           # p=767, 5 paths, 3
    "H100":  [["tsubame/h100_subpipe_repeat_8650141"],   # p=7/15/31, 6, 3
              ["tsubame/h100_subpipe_repeat_8650255"],   # p=255, 6, 3
              ["tsubame/h100_subpipe_repeat_8650143"],   # p=63/511/575, 4, 3
              ["tsubame/h100_subpipe_repeat_8650257",    # p=767, 5, 2+1+1
               "tsubame/h100_subpipe_repeat_8650967",
               "tsubame/h100_subpipe_repeat_8652177"],
              ["tsubame/h100_subpipe_repeat_8631255"]],  # p=63/511/575/767, 4, 2
    "A100":  [["wisteria/a100_subpipe_repeat_9592751"],  # p=7/15/31, 6, 3
              ["wisteria/a100_subpipe_repeat_9593022"],  # p=63, 6, 3
              ["wisteria/a100_subpipe_repeat_9593023"],  # p=127, 6, 3
              ["wisteria/a100_subpipe_repeat_9592752"],  # p=63/511/575, 4, 3
              ["wisteria/a100_subpipe_repeat_9579033"]], # p=63/511/575, 4, 2-3
}
# W2, balance axis: the MPS sweep's own rounds (GB200 3, A100 2; section
# 5AE.1).  The job id is the CSV row's `job`; the directory is found by glob.
BAL_ROUNDS = ["mps_balance_%s", "*/a100_mps_balance_%s", "*/h100_mps_balance_%s"]


def cell(axis, tag, ratio, rp, rm, ledger=True):
    # The leave-one-out GROUP (U1).  The report defines eps_family as "the
    # median error over the OTHER hold-out directions, or on the balance axis
    # over the OTHER machines", and claims that no information about the
    # target machine enters the decision.  The group is therefore the hold-out
    # GPU on the machine axis and the machine on the balance axis -- and in
    # both cases that is the first token of the tag ("GB200 p=511",
    # "A100 p=511 50%").  The axis is carried too, so that the combined
    # both-axes call does not merge a machine's machine-axis cells with its
    # balance-axis cells.
    #
    # V2: `ledger` is False when either path of the ratio had a kernel-ledger
    # mismatch, i.e. when predict_case summed pred and meas over the
    # INTERSECTION of the calibration and target ledgers.  Such a ratio is a
    # ratio of partial sums: it compares the two paths on the subset of kernels
    # both ledgers agree on, which is not the question "which path is faster".
    #
    # W1: err_log is |ln(rp/rm)|, the orientation-free counterpart of err.
    return dict(axis=axis, tag=tag, ratio=ratio, rp=rp, rm=rm, grp=tag.split()[0],
                err=abs(rp / rm - 1.0) * 100.0, hit=(rp > 1) == (rm > 1),
                err_log=abs(math.log(rp / rm)), ledger=bool(ledger))


def direction_cells(srcs, tgt, grp=None, ratios=None):
    """Ratio cells of one transfer direction.  `grp` overrides the leave-one-out
    group, which the isolated pools need: a GB200 -> A100 transfer contributes
    to H100's pool, and there it has to carry H100's group tag so that the
    pool is keyed by the machine being SCORED.  `ratios` defaults to the 7
    published ones (W3 passes ALL_PAIRS)."""
    got, led, roles = {}, {}, {}
    for p in sp.ORDERS:
        for path in sp.LOGO_PATHS:
            pr, me, rows, ok = sp.predict_case(srcs, tgt, p, path)
            if pr is None or not me:
                continue
            got.setdefault(p, {})[path] = (pr, me)
            led[(p, path)] = ok
            # W2: the kernel ledger r_meas was summed over, so that the width
            # can be taken on the same quantity.
            roles[(p, path)] = [r["role"] for r in rows]
    cells = []
    for num, den in ratios or sp.RATIOS:
        for p, r in sorted(got.items()):
            if num in r and den in r:
                c = cell("machine", "%s p=%d" % (grp or tgt, p),
                         num + " / " + den,
                         r[num][0] / r[den][0],
                         r[num][1] / r[den][1],
                         ledger=led[(p, num)] and led[(p, den)])
                c["mach"], c["order"] = tgt, p
                c["roles"] = (roles[(p, num)], roles[(p, den)])
                cells.append(c)
    return cells


def machine_axis(ratios=None):
    cells = []
    for srcs, tgt in LOGO:
        cells += direction_cells(srcs, tgt, ratios=ratios)
    return cells


def isolated_pools(ratios=None):
    """V3.  grp -> the cells of the two single-machine transfers between the
    OTHER two machines.  Used only as the family-error denominator; these cells
    are never scored themselves."""
    out = {}
    for tgt, dirs in SINGLE_POOL.items():
        cs = []
        for srcs, t in dirs:
            cs += direction_cells(srcs, t, grp=tgt, ratios=ratios)
        out[tgt] = cs
    return out


def balance_axis(files=None, jobs=None, ratios=None):
    meas, sm_of, seen, job_of = {}, {}, {}, {}
    for fn in files or [BAL_TIMES]:
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                # subpipe_balance_match.csv holds several jobs of the same
                # cells (the 44 SM point was re-run three times, and 100% was
                # re-measured after it, report section 5AG.3.2b/3.2c/3.5), so
                # it can only be scored one job at a time: --bal-jobs picks
                # which.  Without the filter the duplicate check below stops.
                if jobs and r["job"] not in jobs:
                    continue
                key = (r["machine"], int(r["order"]), int(r["pct"]))
                # One measurement per cell.  A second job at the same
                # (machine, order, pct, path) -- the 2026-09-15 match job
                # re-ran 40% and 100% beside 41% -- must not silently replace
                # the campaign the published score was taken on, so it lives
                # in its own file (subpipe_balance_match.csv) and is scored
                # only when asked for with --bal-times.
                if (key, r["path"]) in seen:
                    sys.exit("duplicate balance cell %s %s in %s (already from "
                             "job %s)" % (key, r["path"], fn, seen[(key, r["path"])]))
                seen[(key, r["path"])] = r["job"]
                meas.setdefault(key, {})[r["path"]] = float(r["device_s_median"])
                sm_of[key] = int(r["sm"])
                job_of[key] = r["job"]
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
            p, _, _, _ = sp.predict_case([mach], key, order, path)
            if p:
                pred.setdefault((mach, order, pct), {})[path] = p
    cells = []
    for k in sorted(meas):
        if k[2] == 100:                          # the calibration point itself
            continue
        for num, den in ratios or sp.RATIOS:
            m, q = meas[k], pred.get(k, {})
            if num in m and den in m and num in q and den in q:
                # ledger=True, and vacuously so: the pseudo target has no
                # sweep of its own, so there is nothing to intersect with and
                # pred is the calibration machine's full ledger, while meas is
                # the production device time of the whole path.  Both sides are
                # path totals, which is what the ledger flag is there to
                # guarantee -- but it is not evidence the way a machine-axis
                # ledger_ok is.
                c = cell("balance", "%s p=%d %d%%" % (k[0], k[1], k[2]),
                         num + " / " + den, q[num] / q[den], m[num] / m[den])
                c["mach"], c["order"], c["pct"], c["job"] = k[0], k[1], k[2], job_of[k]
                cells.append(c)
    return cells


def set_famerr(cells, pool=None):
    """Write fam_loo / sn_post / sn_prior on every cell of `cells`.

    U1.  Until 2026-09-12 the family error was the median over every cell of
    the same ratio with the ONE cell being scored removed:

        fam[c["ratio"]].append(c["err"])
        rest = list(fam[c["ratio"]]); rest.remove(c["err"])

    That leaves the other orders of the SAME hold-out direction in the
    denominator, i.e. the target GPU's own measured errors, which is exactly
    what the rule's statement in section 5W.2 says it does not use.  The
    leave-one-out unit has to be the group the rule holds out: the hold-out
    GPU on the machine axis, the machine on the balance axis.

    V3.  That is still not isolation: with two-machine calibration, the OTHER
    hold-out directions all use the target machine as a calibration source.
    `pool` is the --famerr isolated denominator -- grp -> cells of the
    single-machine transfers between the other two machines, in which the
    scored machine appears nowhere.  A cell whose group has no pool entry for
    its ratio falls back to the leave-one-out group median, and the caller is
    told how many did.

    W1.  fam_log / sn_log are computed beside fam_loo / sn_prior from the SAME
    pool (or the same leave-one-out group when the pool has no entry), with
    |ln(rp/rm)| in place of |rp/rm - 1| and |ln rp| in place of |rp - 1|.
    Which pool a cell uses is decided once, so a cell is never isolated in one
    column and loo-group in the other.
    """
    fam = collections.defaultdict(list)
    for c in cells:
        fam[(c["axis"], c["ratio"])].append((c["grp"], c["err"], c["err_log"]))
    pfam = collections.defaultdict(list)
    if pool:
        keep = {(c["axis"], c["grp"], c["ratio"]) for c in cells}
        for grp, pcells in pool.items():
            for c in pcells:
                if (c["axis"], grp, c["ratio"]) in keep:
                    pfam[(c["axis"], grp, c["ratio"])].append(
                        (c["err"], c["err_log"]))
    n_fallback = 0
    for c in cells:
        key = (c["axis"], c["grp"], c["ratio"])
        if pool and pfam.get(key):
            c["fam_loo"] = st.median([e for e, _ in pfam[key]])
            c["fam_log"] = st.median([e for _, e in pfam[key]])
        else:
            rest = [(e, el) for g, e, el in fam[(c["axis"], c["ratio"])]
                    if g != c["grp"]]
            c["fam_loo"] = st.median([e for e, _ in rest]) if rest else c["err"]
            c["fam_log"] = (st.median([e for _, e in rest]) if rest
                            else c["err_log"])
            # Only count it as a fallback if the pool covers this axis at all;
            # the balance axis has no isolated pool by construction.
            n_fallback += bool(pool) and any(
                pc["axis"] == c["axis"] for pcs in pool.values() for pc in pcs)
        c["sn_post"] = (abs(c["rm"] - 1) * 100 / c["err"]) if c["err"] else 9e9
        c["sn_prior"] = abs(c["rp"] - 1) * 100 / c["fam_loo"]
        c["sn_log"] = (abs(math.log(c["rp"])) / c["fam_log"] if c["fam_log"]
                       else 9e9)
    return n_fallback


def score(cells, label, pool=None, why=True):
    if not cells:
        print("\n=== %s: no cells" % label)
        return
    n_fb = set_famerr(cells, pool)
    print("\n=== %s: %d cells, %d ranked right (%.0f%%)%s"
          % (label, len(cells), sum(c["hit"] for c in cells),
             100 * sum(c["hit"] for c in cells) / len(cells),
             "  [%d cells fell back to the loo-group family error]" % n_fb
             if n_fb else ""))
    for key, name in (("sn_post", "posterior S/N (published: uses r_meas)"),
                      ("sn_prior", "prior S/N (|r_pred-1| / family error)"),
                      ("sn_log", "log S/N (|ln r_pred| / median |ln err|, W1)")):
        print("  %s" % name)
        for t in (1.0, 1.1, 1.5, 2.0):
            hi = [c for c in cells if c[key] >= t]
            lo = [c for c in cells if c[key] < t]
            print("    t=%.1f  >=t %3d/%3d = %5.1f%%   <t %3d/%3d = %5.1f%%"
                  % (t, sum(c["hit"] for c in hi), len(hi),
                     100 * sum(c["hit"] for c in hi) / max(1, len(hi)),
                     sum(c["hit"] for c in lo), len(lo),
                     100 * sum(c["hit"] for c in lo) / max(1, len(lo))))
    # The threshold scan the report quotes a minimum from.
    worst = min(((t, [c for c in cells if c["sn_prior"] >= t])
                 for t in [1.0 + 0.05 * i for i in range(41)]),
                key=lambda tc: (100 * sum(c["hit"] for c in tc[1])
                                / max(1, len(tc[1]))))
    print("    prior S/N worst t in [1.0, 3.0]: t=%.2f  %d/%d = %.1f%%"
          % (worst[0], sum(c["hit"] for c in worst[1]), len(worst[1]),
             100 * sum(c["hit"] for c in worst[1]) / max(1, len(worst[1]))))
    # Per leave-one-out group at the published threshold, so that the
    # three hold-out GPUs (or the two MPS machines) can be read off.
    grps = sorted({c["grp"] for c in cells})
    if len(grps) > 1:
        print("    per group at t=1.1 (prior): "
              + ", ".join("%s %d/%d" % (g, sum(c["hit"] for c in h), len(h))
                          for g in grps
                          for h in [[c for c in cells
                                     if c["grp"] == g and c["sn_prior"] >= 1.1]]))
    # Order bands, at the published threshold.  The paper scores the two low
    # orders separately ("27 of 28 resolved"), and V2 matters most there: every
    # kernel-ledger mismatch in the whole study is at p = 7 or p = 15.
    bands = [("p<=15", lambda o: o <= 15), ("p=31..127", lambda o: 31 <= o <= 127),
             ("p>=255", lambda o: o >= 255)]
    print("    per order band at t=1.1 (prior): " + ", ".join(
        "%s %d/%d of %d" % (nm, sum(c["hit"] for c in h), len(h), len(b))
        for nm, f in bands
        for b in [[c for c in cells if f(int(c["tag"].split("p=")[1].split()[0]))]]
        for h in [[c for c in b if c["sn_prior"] >= 1.1]] if b))
    if not why:
        return
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


def keep_axis(cells, key="axis"):
    """Freeze the per-axis family error before score() is called again.

    score() recomputes fam_loo over whatever set it is given, so the final
    both-axes call would otherwise overwrite the per-axis values that the
    figure is scored with.
    """
    for c in cells:
        c["fam_" + key], c["sn_" + key] = c["fam_loo"], c["sn_prior"]
        c["famlog_" + key], c["snlog_" + key] = c["fam_log"], c["sn_log"]


def score_quiet(cells, pool=None):
    """The S/N columns of report_axis() without the printing: fam_axis /
    sn_axis over all cells, fam_led / sn_led over the ledger-consistent ones
    (each with its pool restricted the same way), and the all-cells values
    restored afterwards.  W3 uses it for the all-pairs dump."""
    set_famerr(cells, pool)
    keep_axis(cells, "axis")
    ok = [c for c in cells if c["ledger"]]
    pool_ok = ({g: [c for c in cs if c["ledger"]] for g, cs in pool.items()}
               if pool else None)
    set_famerr(ok, pool_ok)
    keep_axis(ok, "led")
    set_famerr(cells, pool)


_ROUND_CACHE = {}


def round_totals(d, order, path, rnd):
    """W2.  {role: median launch duration} of one round of one repeat job,
    with the roles split into occurrences exactly as load_case_raw does, so
    that the role names match the scored ledger.  None when the round has no
    file.  Only the _a pass is read: it carries `dur`."""
    key = (d, order, path, rnd)
    if key in _ROUND_CACHE:
        return _ROUND_CACHE[key]
    files = sorted(glob.glob(os.path.join(sp.ROOT, d, "ncu",
                                          "p%d_%s_8x8x4_none_r%s_a.csv"
                                          % (order, path, rnd))))
    per_role, seq = collections.defaultdict(list), []
    for f in files:
        for _, (role, mets) in sorted(sp.read_csv(f).items(),
                                      key=lambda kv: int(kv[0])):
            if role and "dur" in mets:
                per_role[role].append(mets)
                seq.append(role)
    out = None
    if per_role:
        per_role = sp.split_occurrences(per_role, sp.launch_multiplicity(seq))
        out = {r: st.median([x["dur"] for x in lst])
               for r, lst in per_role.items()}
    _ROUND_CACHE[key] = out
    return out


def rounds_of(dirs, order, path):
    """(directory, round) keys of one job's rounds of (order, path)."""
    out = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(sp.ROOT, d, "ncu",
                                               "p%d_%s_8x8x4_none_r*_a.csv"
                                               % (order, path)))):
            out.append((d, os.path.basename(f).split("_r")[-1].split("_")[0]))
    return out


def job_name(dirs):
    return "+".join(d.rstrip("/").split("/")[-1].split("_")[-1] for d in dirs)


def width_of(vals):
    return (max(vals) - min(vals)) / min(vals) * 100.0


def set_widths(cells):
    """W2.  Write width / resolved / width_job on every cell that a repeat job
    covers.  Machine axis: the ratio of the two paths' per-round totals, each
    summed over the cell's scored ledger (c["roles"]); a round that lacks a
    scored role is unusable and a job with fewer than two usable rounds common
    to both paths is skipped.  Balance axis: the ratio of the two paths'
    per-round `CUDA device` sums from the MPS job's logs, the same quantity
    subpipe_balance_distill takes the median of."""
    n = 0
    for c in cells:
        c["width"] = c["width_job"] = None
        num, den = [x.strip() for x in c["ratio"].split("/")]
        rr = None
        if c["axis"] == "machine":
            rn, rd = c["roles"]
            for dirs in REPEATS.get(c["mach"], []):
                tot = {}
                for path, roles in ((num, rn), (den, rd)):
                    tot[path] = {}
                    for rk in rounds_of(dirs, c["order"], path):
                        t = round_totals(rk[0], c["order"], path, rk[1])
                        if t and all(r in t for r in roles):
                            tot[path][rk] = sum(t[r] for r in roles)
                common = sorted(set(tot[num]) & set(tot[den]))
                if len(common) >= 2:
                    rr = [tot[num][k] / tot[den][k] for k in common]
                    c["width_job"] = job_name(dirs)
                    break
        else:
            outdir = None
            for pat in BAL_ROUNDS:
                hit = sorted(glob.glob(os.path.join(sp.ROOT, pat % c["job"])))
                if hit:
                    outdir = hit[0]
                    break
            if outdir:
                import subpipe_balance_distill as bd
                tot = {}
                for path in (num, den):
                    tot[path] = {}
                    for log in glob.glob(os.path.join(
                            outdir, "logs", "r*_p%d_%dpct_%s.log"
                            % (c["order"], c["pct"], path))):
                        t = bd.device_sum(log)
                        if t is not None:
                            tot[path][os.path.basename(log).split("_")[0]] = t
                common = sorted(set(tot[num]) & set(tot[den]))
                if len(common) >= 2:
                    rr = [tot[num][k] / tot[den][k] for k in common]
                    c["width_job"] = c["job"]
        if rr is not None:
            c["width"] = width_of(rr)
            c["width_n"] = len(rr)
            c["resolved"] = abs(c["rm"] - 1) * 100.0 > c["width"]
            n += 1
    return n


def report_widths(cells, label):
    have = [c for c in cells if c.get("width") is not None]
    print("\n=== %s: run-to-run width on %d of %d cells" % (label, len(have),
                                                             len(cells)))
    if not have:
        return
    w = [c["width"] for c in have]
    print("    width median %.3f%% / max %.3f%%; resolved (|r_meas-1| > width) "
          "%d/%d" % (st.median(w), max(w), sum(c["resolved"] for c in have),
                     len(have)))
    by = collections.Counter(c["width_job"] for c in have)
    print("    jobs: " + ", ".join("%s x%d" % kv for kv in sorted(by.items())))
    miss = collections.Counter(c["tag"].split(" ")[0] + " " + c["tag"].split(" ")[1]
                               for c in cells if c.get("width") is None)
    if miss:
        print("    no width: " + ", ".join("%s x%d" % kv
                                          for kv in sorted(miss.items())))


def report_axis(cells, label, pool=None):
    """V2: score the path-total cells and the partial-sum cells separately.

    Both numbers are printed.  The ledger-consistent subset is the one that
    answers "which path is faster"; the mismatched cells answer "which path is
    faster on the kernels both ledgers agree on", which is a different and
    weaker question, so they are scored but kept apart.  The family error of
    each subset is computed within that subset (and, with `pool`, from the
    isolated pool restricted the same way), so neither borrows the other's
    error scale.
    """
    ok = [c for c in cells if c["ledger"]]
    bad = [c for c in cells if not c["ledger"]]
    score(cells, label + " -- ALL cells", pool)
    keep_axis(cells, "axis")                    # the dump's default S/N
    if bad:
        pool_ok = ({g: [c for c in cs if c["ledger"]]
                    for g, cs in pool.items()} if pool else None)
        score(ok, label + " -- ledger-consistent cells only (path totals)",
              pool_ok, why=False)
        keep_axis(ok, "led")
        score(bad, label + " -- kernel-ledger mismatch (common-subset "
                            "comparison, NOT a path total)", pool, why=False)
        by = collections.Counter(c["tag"] for c in bad)
        print("    the mismatched cells: "
              + ", ".join("%s x%d" % (k, v) for k, v in sorted(by.items())))
        set_famerr(cells, pool)                 # restore the all-cells values
    else:
        print("  (every cell of this axis is ledger-consistent)")
        keep_axis(ok, "led")
    if pool:
        pe = [c["err"] for cs in pool.values() for c in cs]
        pl = [c["err_log"] for cs in pool.values() for c in cs]
        ce = [c["err"] for c in cells]
        cl = [c["err_log"] for c in cells]
        print("  CONSTRAINT of --famerr isolated: the denominator is a "
              "single-machine calibration, a different configuration.  Its "
              "|ratio err| median is %.2f%% over %d cells, against %.2f%% "
              "over the %d two-machine cells being scored (%.2fx); in the "
              "log form the paper scores on (W1), median |ln err| is %.2f%% "
              "against %.2f%% (%.2fx)."
              % (st.median(pe), len(pe), st.median(ce), len(cells),
                 st.median(pe) / st.median(ce),
                 100 * st.median(pl), 100 * st.median(cl),
                 st.median(pl) / st.median(cl)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="both",
                    choices=["machine", "balance", "both"])
    ap.add_argument("--from-table", action="store_true",
                    help="read counters from analysis/subpipe_inputs.csv")
    ap.add_argument("--dump-cells",
                    help="write one row per scored cell to this CSV, in the "
                         "schema the paper's Figure 8 reads (figs/data/"
                         "sn_cells.csv): axis, machine, order, the two paths, "
                         "the predicted and measured ratio, the leave-one-out "
                         "family error as a fraction, the a priori S/N and the "
                         "hit.  The MPS fraction of a balance cell is not "
                         "carried: the figure does not distinguish them.")
    ap.add_argument("--dump-all-pairs",
                    help="W3: write the same schema as --dump-cells for ALL "
                         "15 pairs of the six transferred paths (ALL_PAIRS), "
                         "both axes as selected by --axis, with the family "
                         "error of each pair from its own pool.  The rows of "
                         "the 7 published pairs are identical to --dump-cells.")
    ap.add_argument("--no-width", action="store_true",
                    help="skip W2 (the repeat-job scan); width columns empty")
    ap.add_argument("--shape", default="static",
                    choices=["static", "static-only", "target"],
                    help="passed through to subpipe_predict.SHAPE_SRC; see "
                         "its --shape")
    ap.add_argument("--famerr", default="loo-group",
                    choices=["loo-group", "isolated"],
                    help="what the prior S/N divides by.  loo-group: the "
                         "median error of the OTHER hold-out groups of the "
                         "same ratio (the rule as stated in section 5W.2).  "
                         "isolated: the median error of the single-machine "
                         "transfers between the other two machines, in which "
                         "the scored machine appears neither as target nor as "
                         "calibration source (V3).  Machine axis only -- the "
                         "balance axis's groups are already isolated, since a "
                         "machine's family error there comes from the other "
                         "machine's MPS sweep")
    ap.add_argument("--bal-times", default=BAL_TIMES,
                    help="comma-separated CSVs of balance-axis times (default: "
                         "the two-machine sweep, analysis/subpipe_balance_times"
                         ".csv).  The balance-matched job is in "
                         "analysis/subpipe_balance_match.csv and is NOT part "
                         "of the published 190-cell score; pass both to score "
                         "the 41%% column as well.")
    ap.add_argument("--bal-jobs", default="",
                    help="comma-separated job ids to keep from the balance "
                         "CSVs; rows of any other job are ignored.  Needed "
                         "with analysis/subpipe_balance_match.csv, which "
                         "holds the same cells from several jobs and is "
                         "otherwise refused as duplicate (section 5AG.3.3)")
    a = ap.parse_args()
    sp.USE_TABLE[0] = a.from_table
    sp.SHAPE_SRC[0] = a.shape
    FAMERR[0] = a.famerr
    cells, m, pool = [], [], None
    if a.axis in ("machine", "both"):
        m = machine_axis()
        pool = isolated_pools() if a.famerr == "isolated" else None
        report_axis(m, "machine axis (leave-one-GPU-out, famerr=%s)" % a.famerr,
                    pool)
        cells += m
    if a.axis in ("balance", "both"):
        b = balance_axis(a.bal_times.split(","),
                         set(a.bal_jobs.split(",")) - {""})
        report_axis(b, "balance axis (MPS SM fraction)")
        cells += b
    if a.axis == "both" and len(cells) > len(m):
        score(cells, "both axes -- ALL cells", pool)
        ok = [c for c in cells if c["ledger"]]
        if len(ok) < len(cells):
            pool_ok = ({g: [c for c in cs if c["ledger"]]
                        for g, cs in pool.items()} if pool else None)
            score(ok, "both axes -- ledger-consistent cells only", pool_ok,
                  why=False)
    if not a.no_width and (a.dump_cells or a.dump_all_pairs):
        set_widths(cells)
        for ax in ("machine", "balance"):
            sub = [c for c in cells if c["axis"] == ax]
            if sub:
                report_widths(sub, ax + " axis")
    if a.dump_cells:
        dump(a.dump_cells, cells)
    if a.dump_all_pairs:
        # W3.  Rebuilt from scratch with ALL_PAIRS; the scoring is the quiet
        # twin of what report_axis did above, per axis, with the same pool
        # convention, so the 7 published pairs come out identical.
        allc = []
        if a.axis in ("machine", "both"):
            ma = machine_axis(ALL_PAIRS)
            score_quiet(ma, isolated_pools(ALL_PAIRS)
                        if a.famerr == "isolated" else None)
            allc += ma
        if a.axis in ("balance", "both"):
            ba = balance_axis(a.bal_times.split(","),
                              set(a.bal_jobs.split(",")) - {""}, ALL_PAIRS)
            score_quiet(ba)
            allc += ba
        if not a.no_width:
            set_widths(allc)
        dump(a.dump_all_pairs, allc)


def dump(fn, cells):
    """--dump-cells schema.  The first 13 columns are the published ones and
    are written exactly as before; W1 / W2 add columns at the END only.
    famerr_log is the median |ln(r_pred/r_meas)| itself (already
    dimensionless, unlike famerr, which is a fraction of 1); width is in
    percent; resolved is 1/0, empty when there is no width; pct is the MPS
    fraction of a balance cell (the published columns do not carry it, so
    the five fractions of one (machine, order, ratio) were told apart only by
    row order) and empty on the machine axis."""
    with open(fn, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["axis", "grp", "order", "num", "den", "pred", "meas",
                    "famerr", "sn", "hit", "ledger_ok", "famerr_led",
                    "sn_led",
                    "famerr_log", "sn_log", "famerr_led_log", "sn_led_log",
                    "width", "resolved", "width_job", "pct"])
        for c in cells:
            grp = c["tag"].split()[0]
            order = int(c["tag"].split("p=")[1].split()[0])
            num, den = [x.strip() for x in c["ratio"].split("/")]
            w.writerow([c["axis"], grp, order, num, den,
                        "%.4f" % c["rp"], "%.4f" % c["rm"],
                        "%.4f" % (c["fam_axis"] / 100.0),
                        "%.4f" % c["sn_axis"], int(c["hit"]),
                        int(c["ledger"]),
                        "%.4f" % (c["fam_led"] / 100.0)
                        if "fam_led" in c else "",
                        "%.4f" % c["sn_led"] if "sn_led" in c else "",
                        "%.4f" % c["famlog_axis"], "%.4f" % c["snlog_axis"],
                        "%.4f" % c["famlog_led"] if "famlog_led" in c else "",
                        "%.4f" % c["snlog_led"] if "snlog_led" in c else "",
                        "%.3f" % c["width"] if c.get("width") is not None else "",
                        int(c["resolved"]) if c.get("width") is not None else "",
                        c.get("width_job") or "", c.get("pct", "")])
    print("\nwrote %s (%d cells)" % (fn, len(cells)))


if __name__ == "__main__":
    main()
