#!/usr/bin/env python3
"""Turn a jobs/tsubame_paths.sh log into a median table per (order, path).

Usage:  python3 jobs/tsubame_paths_summarize.py <job stdout log> [--csv out.csv]

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
        })

    width = max(len(r["path"]) for r in rows)
    print("%-5s %-*s %5s %5s %14s %10s %7s %7s %12s"
          % ("p", width, "path", "runs", "fail", "main ms/step", "min", "med/min",
             "clean", "stage us"))
    last_order = None
    for row in rows:
        if last_order is not None and row["poly_order"] != last_order:
            print()
        last_order = row["poly_order"]
        flag = "*" if row["clean_runs"] != "" and row["clean_runs"] * 2 < row["runs"] else " "
        print("%-5d %-*s %5d %5d %14s %10s %6s%%%s %6s %12s" % (
            row["poly_order"], width, row["path"], row["runs"], row["failed"],
            "%.4f" % row["main_ms_per_step_median"] if row["main_ms_per_step_median"] != "" else "-",
            "%.4f" % row["main_ms_per_step_min"] if row["main_ms_per_step_min"] != "" else "-",
            "%.2f" % row["median_over_min_pct"] if row["median_over_min_pct"] != "" else "-",
            flag,
            "%d/%d" % (row["clean_runs"], row["runs"]) if row["clean_runs"] != "" else "-",
            "%.2f" % row["stage_us_median"] if row["stage_us_median"] != "" else "-"))

    # Fastest path per order, on the same axis the GB200 summary table is bold on.
    print()
    print("fastest per order (median main ms/step):")
    by_order: dict[int, list] = {}
    for row in rows:
        if row["main_ms_per_step_median"] != "":
            by_order.setdefault(row["poly_order"], []).append(row)
    for order in sorted(by_order):
        best = min(by_order[order], key=lambda r: r["main_ms_per_step_median"])
        print("  p=%-5d %-*s %.4f" % (order, width, best["path"],
                                      best["main_ms_per_step_median"]))

    starred = [r for r in rows
               if r["clean_runs"] != "" and r["clean_runs"] * 2 < r["runs"]]
    if starred:
        print()
        print("* fewer than half the rounds are within 0.5%% of the min -- "
              "interference landed on this cell, quote the min:")
        for row in starred:
            print("    p=%-5d %-*s median %.4f  min %.4f  (%+.2f%%)" % (
                row["poly_order"], width, row["path"],
                row["main_ms_per_step_median"], row["main_ms_per_step_min"],
                row["median_over_min_pct"]))

    if failures:
        print()
        print("failed runs (%d):" % len(failures))
        for text in sorted(set(failures)):
            print("  " + text)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("\nwrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
