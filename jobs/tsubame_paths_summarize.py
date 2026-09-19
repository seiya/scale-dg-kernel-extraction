#!/usr/bin/env python3
"""Turn a jobs/tsubame_paths.sh log into a median table per (order, path).

Usage:  python3 jobs/tsubame_paths_summarize.py <job stdout log> [--csv out.csv]

The PRIMARY axis is "Step loop per stage".  Main per step is reported beside
it as the end-to-end cross-check AGENTS.md asks for, but it is not the
cross-order axis: Main's window contains the in-loop diagnostic (the q extrema
reduction and its print), and output_interval differs by order, so the number
of diagnostics inside the measured window is an order-dependent contaminant.
Step loop brackets those out, from device sync to device sync.

Measured, so that the change of axis is not a claim: on Slurm 81583 and SGE
8567869 the two agree to within 0.17% everywhere and the cross-order H100/GB200
ratio table is identical to three digits (reports/h100_report.md 8.9.1).  What
the change does buy is robustness to interference -- at p=31 on H100 the two
medians disagree by 6.0% on one cell, because the outlier landed in Main's
median and not in stage's.

Reports the MEDIAN over the interleaved rounds: the rounds exist to absorb
clock and neighbour drift, and interleaving is what makes the median fair
across paths.  Failed runs (rc != 0) are counted and listed, not silently
dropped -- at p=767 an OOM on a 94 GB H100 is a result.

The min and "clean" columns are the guard on that.  Interference that lands on
whole runs adds a roughly constant offset and hits some paths more often than
others, and then the median of six rounds is NOT protected: on TSUBAME job
8567869 exactly one cell (p=31 GEMM_CUTE@16x8x4) took the hit in four rounds
out of six and its median came out 5.93% above its min, while every other cell
on that machine was within 1.6% and the whole GB200 job within 0.38%.  "clean"
counts the rounds within 0.5% of the min; when it is a minority, quote the min
and say why, or take the order again.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys

ROW = re.compile(
    r"^r(\d+)\s+p(\d+)\s+(\S+)\s+rc=(\S+)\s+steps=(\S*)\s+devsum=(\S+)\s+"
    r"stage=(\S*)\s+main=(\S*)\s*$"
)


def as_float(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--csv")
    args = parser.parse_args()

    samples: dict[tuple[int, str], dict[str, list]] = {}
    failures: list[str] = []
    for line in open(args.log, errors="replace"):
        match = ROW.match(line.strip())
        if not match:
            continue
        _round, order, path, rc, _steps, devsum, stage, main = match.groups()
        key = (int(order), path)
        entry = samples.setdefault(key, {"main": [], "stage": [], "dev": [], "fail": 0})
        if rc != "0":
            entry["fail"] += 1
            failures.append("p=%s %s rc=%s" % (order, path, rc))
            continue
        for name, text in (("main", main), ("stage", stage), ("dev", devsum)):
            value = as_float(text)
            if value is not None:
                entry[name].append(value)

    if not samples:
        print("no measurement rows found in %s" % args.log, file=sys.stderr)
        return 1

    rows = []
    for (order, path), entry in sorted(samples.items()):
        main_ms = [1e3 * v for v in entry["main"]]
        stage_us = [1e6 * v for v in entry["stage"]]
        rows.append({
            "poly_order": order,
            "path": path,
            "runs": len(main_ms),
            "failed": entry["fail"],
            "main_ms_per_step_median": statistics.median(main_ms) if main_ms else "",
            "main_ms_per_step_min": min(main_ms) if main_ms else "",
            "median_over_min_pct": (
                100.0 * (statistics.median(main_ms) / min(main_ms) - 1.0)
                if main_ms else ""),
            "clean_runs": (
                sum(1 for v in main_ms if v <= min(main_ms) * 1.005)
                if main_ms else ""),
            "main_ms_spread_pct": (
                100.0 * (max(main_ms) - min(main_ms)) / statistics.median(main_ms)
                if len(main_ms) > 1 else ""),
            "stage_us_median": statistics.median(stage_us) if stage_us else "",
            "stage_us_min": min(stage_us) if stage_us else "",
            "stage_us_max": max(stage_us) if stage_us else "",
            "stage_median_over_min_pct": (
                100.0 * (statistics.median(stage_us) / min(stage_us) - 1.0)
                if stage_us else ""),
            # Median absolute deviation, as a percentage of the median.  Unlike
            # the spread it does not move when one round out of many is hit, so
            # the pair (MAD, spread) separates "noisy" from "one bad round".
            "stage_mad_pct": (
                100.0 * statistics.median([abs(v - statistics.median(stage_us))
                                           for v in stage_us])
                / statistics.median(stage_us)
                if stage_us else ""),
            "stage_us_spread_pct": (
                100.0 * (max(stage_us) - min(stage_us)) / statistics.median(stage_us)
                if len(stage_us) > 1 else ""),
            "stage_clean_runs": (
                sum(1 for v in stage_us if v <= min(stage_us) * 1.005)
                if stage_us else ""),
        })

    width = max(len(r["path"]) for r in rows)
    print("PRIMARY AXIS: Step loop per stage (us).  main ms/step is the "
          "end-to-end cross-check.")
    print("%-5s %-*s %5s %5s %12s %10s %7s %7s %7s %14s"
          % ("p", width, "path", "runs", "fail", "stage us", "min", "med/min",
             "MAD", "clean", "main ms/step"))
    last_order = None
    for row in rows:
        if last_order is not None and row["poly_order"] != last_order:
            print()
        last_order = row["poly_order"]
        flag = ("*" if row["stage_clean_runs"] != ""
                and row["stage_clean_runs"] * 2 < row["runs"] else " ")
        def show(key, fmt="%.2f"):
            return fmt % row[key] if row[key] != "" else "-"
        print("%-5d %-*s %5d %5d %12s %10s %6s%%%s %6s%% %6s %14s" % (
            row["poly_order"], width, row["path"], row["runs"], row["failed"],
            show("stage_us_median"), show("stage_us_min"),
            show("stage_median_over_min_pct"), flag, show("stage_mad_pct"),
            "%d/%d" % (row["stage_clean_runs"], row["runs"])
            if row["stage_clean_runs"] != "" else "-",
            show("main_ms_per_step_median", "%.4f")))

    # Fastest path per order, on the same axis the GB200 summary table is bold on.
    print()
    print("fastest per order (median Step loop per stage):")
    by_order: dict[int, list] = {}
    for row in rows:
        if row["stage_us_median"] != "":
            by_order.setdefault(row["poly_order"], []).append(row)
    for order in sorted(by_order):
        best = min(by_order[order], key=lambda r: r["stage_us_median"])
        alt = min(by_order[order], key=lambda r: r["main_ms_per_step_median"]
                  if r["main_ms_per_step_median"] != "" else float("inf"))
        note = "" if alt["path"] == best["path"] else (
            "   (main would pick %s -- interference, see the docstring)" % alt["path"])
        print("  p=%-5d %-*s %.2f us/stage%s" % (order, width, best["path"],
                                                 best["stage_us_median"], note))

    starred = [r for r in rows
               if r["stage_clean_runs"] != ""
               and r["stage_clean_runs"] * 2 < r["runs"]]
    if starred:
        print()
        print("* fewer than half the rounds are within 0.5% of the min -- "
              "interference landed on this cell, quote the min:")
        for row in starred:
            print("    p=%-5d %-*s median %.2f us  min %.2f us  (%+.2f%%)" % (
                row["poly_order"], width, row["path"],
                row["stage_us_median"], row["stage_us_min"],
                row["stage_median_over_min_pct"]))

    if failures:
        print()
        print("failed runs (%d):" % len(failures))
        for text in sorted(set(failures)):
            print("  " + text)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print("\nwrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
