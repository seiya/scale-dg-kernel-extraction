#!/usr/bin/env python3
"""Distill one MPS balance job (jobs/mps_balance.sh output) into the rows of
analysis/subpipe_balance_times.csv.

The CSV is what analysis/subpipe_rank_audit.py scores the balance axis on, so
the step from a job's logs to its rows has to be reproducible too.  One row per
(order, percentage, path): the median over rounds of the "CUDA device" sum the
production timers print, and the SM count the percentage actually granted --
read from the job's own smid.txt when it has one (jobs/mps_balance.sh records
it since 2026-09-15), otherwise from --sm, never guessed from the percentage
(reports/subpipe_transfer_model.md 5Z.7: TPC granularity, and A100 gets fewer
than the floor at 75/40/25%).

  python3 analysis/subpipe_balance_distill.py MACHINE JOB OUTDIR [--sm 41=44 ...]
      [--csv analysis/subpipe_balance_times.csv] [--write] [--check]

Without --write it prints the rows.  --write merges them into the CSV,
replacing any row with the same (machine, job, order, pct, path).  --check
compares them to the rows already in the CSV for that job and exits non-zero
on any difference, which is how the script was validated against the rows
distilled by hand for GB200 96451 and A100 9580356.
"""
import argparse
import collections
import csv
import glob
import os
import re
import statistics as st
import sys

LOG_RE = re.compile(r"r(\d+)_p(\d+)_(\d+)pct_([A-Z_0-9]+)\.log$")
CSV_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "subpipe_balance_times.csv")
FIELDS = ["machine", "job", "order", "pct", "sm", "path", "rounds",
          "device_s_median"]


def device_sum(path):
    s, seen = 0.0, False
    with open(path) as f:
        for line in f:
            if "CUDA device" in line:
                s += float(line.split()[-1]); seen = True
    return s if seen and s > 0 else None


def read_smid(outdir):
    sm = {}
    p = os.path.join(outdir, "smid.txt")
    if os.path.exists(p):
        for line in open(p):
            m = re.match(r"smid pct=(\d+)\s+device_sm=\d+ used_sm=(\d+)", line)
            if m:
                sm[int(m.group(1))] = int(m.group(2))
    return sm


def distill(machine, job, outdir, sm):
    runs = collections.defaultdict(list)
    for log in sorted(glob.glob(os.path.join(outdir, "logs", "*.log"))):
        m = LOG_RE.search(os.path.basename(log))
        if not m:
            continue
        _, order, pct, path = m.groups()
        t = device_sum(log)
        if t is not None:                 # rc!=0 runs (p=511 FUSED) have no timer
            runs[(int(order), int(pct), path)].append(t)
    rows = []
    for (order, pct, path), ts in sorted(runs.items()):
        if pct not in sm:
            sys.exit(f"no SM count for {pct}%: no smid.txt in {outdir}, pass --sm {pct}=N")
        rows.append(dict(machine=machine, job=str(job), order=str(order),
                         pct=str(pct), sm=str(sm[pct]), path=path,
                         rounds=str(len(ts)),
                         device_s_median=f"{st.median(ts):.6E}"))
    return rows


def key(r):
    return (r["machine"], r["job"], r["order"], r["pct"], r["path"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("machine"); ap.add_argument("job"); ap.add_argument("outdir")
    ap.add_argument("--sm", action="append", default=[], metavar="PCT=NSM")
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sm = read_smid(a.outdir)
    for kv in a.sm:
        k, v = kv.split("="); sm[int(k)] = int(v)
    rows = distill(a.machine, a.job, a.outdir, sm)
    have = list(csv.DictReader(open(a.csv))) if os.path.exists(a.csv) else []
    if a.check:
        old = {key(r): r for r in have if r["job"] == str(a.job)}
        bad = 0
        for r in rows:
            o = old.get(key(r))
            if o is None:
                print(f"NEW   {key(r)}"); bad += 1
            elif o["device_s_median"] != r["device_s_median"] or o["sm"] != r["sm"] \
                    or o["rounds"] != r["rounds"]:
                print(f"DIFF  {key(r)} csv={o} new={r}"); bad += 1
        print(f"{len(rows)} rows, {bad} not matching the CSV")
        sys.exit(1 if bad else 0)
    if a.write:
        new = {key(r): r for r in rows}
        merged = [new.pop(key(r), r) for r in have] + list(new.values())
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(merged)
        print(f"{len(rows)} rows merged into {a.csv} ({len(merged)} total)")
    else:
        w = csv.DictWriter(sys.stdout, FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
