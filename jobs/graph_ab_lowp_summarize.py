#!/usr/bin/env python3
"""Summarize jobs/graph_ab_lowp.sh output trees (CUDA-graph on/off A/B, p=7, 15).

    python3 jobs/graph_ab_lowp_summarize.py [--csv out.csv] OUTDIR [OUTDIR ...]

Each OUTDIR is one machine's run (metadata.txt names the machine).  Prints,
per (machine, order, path, mma_shape): the round median and MAD of the
step-loop time per RK stage and of the end-to-end time per step, graph off
and on, and the graph's saving Delta = off - on.  Then the rankings the paper
claims at p <= 15 -- the seven ratio families of sn_cells.csv, FUSED_TC/FUSED
of Fig. 1(b), and the fastest path per (machine, order) -- evaluated on the
stage medians with the graph off and on, and every cell whose sign moves.

--csv writes one row per run: machine,order,path,mma_shape,graph,round,
main_per_step_s,steploop_per_stage_s (LF line endings).
"""
import argparse, csv, glob, os, re, statistics, sys

RUN_RE = re.compile(r"perf_r(\d+)_p(\d+)_(CUDAFORTRAN_[A-Z_]+?)(?:_(\d+x\d+x\d+))?_(false|true)\.log$")
RATIOS = [("GEMM_FUSED", "FUSED_TC"), ("GEMM", "FUSED_TC"), ("FUSED_DFMA", "FUSED_TC"),
          ("GEMM_CUTE", "GEMM_FUSED"), ("GEMM_CUTE", "GEMM"), ("FUSED_DFMA", "FUSED"),
          ("FUSED", "FUSED_TC"), ("FUSED_TC", "FUSED")]   # last: Fig. 1(b)


def read_runs(outdir):
    meta = {}
    with open(os.path.join(outdir, "metadata.txt")) as f:
        for line in f:
            if "=" in line and not line.startswith(" "):
                k, v = line.rstrip("\n").split("=", 1)
                meta[k] = v
    machine = meta.get("machine", "?").upper()
    rows = []
    for path in sorted(glob.glob(os.path.join(outdir, "run_logs", "perf_r*.log"))):
        m = RUN_RE.search(os.path.basename(path))
        if not m:
            continue
        r, p, kern, shape, g = m.groups()
        main = stage = None
        with open(path) as f:
            for line in f:
                if "Main per step:" in line:
                    main = float(line.split()[-1])
                elif "Step loop per stage:" in line:
                    stage = float(line.split()[-1])
        if main is None or stage is None:
            print("  skipped (no timing):", path, file=sys.stderr)
            continue
        rows.append(dict(machine=machine, order=int(p), path=kern.replace("CUDAFORTRAN_", ""),
                         mma_shape=shape or "8x8x4", graph=g, round=int(r),
                         main_per_step_s=main, steploop_per_stage_s=stage))
    return meta, rows


def med_mad(xs):
    m = statistics.median(xs)
    return m, statistics.median([abs(x - m) for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdirs", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = []
    for d in a.outdirs:
        meta, rr = read_runs(d)
        print("%s: machine=%s job=%s host=%s commit=%s exe_sha256=%s runs=%d" % (
            d, meta.get("machine"), meta.get("job_id"), meta.get("host"),
            meta.get("commit", "?")[:7], meta.get("exe_sha256", "?")[:12], len(rr)))
        rows += rr
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["machine", "order", "path", "mma_shape", "graph", "round",
                        "main_per_step_s", "steploop_per_stage_s"])
            for r in sorted(rows, key=lambda r: (r["machine"], r["order"], r["path"], r["mma_shape"], r["graph"], r["round"])):
                w.writerow([r["machine"], r["order"], r["path"], r["mma_shape"], r["graph"], r["round"],
                            "%.5E" % r["main_per_step_s"], "%.5E" % r["steploop_per_stage_s"]])
        print("wrote %s (%d rows)" % (a.csv, len(rows)))

    # medians
    cells = {}
    for r in rows:
        cells.setdefault((r["machine"], r["order"], r["path"], r["mma_shape"], r["graph"]), []).append(r)
    med = {}
    print("\n%-6s %4s %-11s %-7s | %14s %14s %14s | %14s %14s %14s" % (
        "mach", "p", "path", "shape", "stage off us", "stage on us", "dStage us (%)",
        "main off us", "main on us", "dMain us (%)"))
    for key in sorted(set((k[0], k[1], k[2], k[3]) for k in cells)):
        off = cells.get(key + ("false",)); on = cells.get(key + ("true",))
        if not off or not on:
            continue
        so, sm = med_mad([r["steploop_per_stage_s"] * 1e6 for r in off])
        s1, s1m = med_mad([r["steploop_per_stage_s"] * 1e6 for r in on])
        mo, mom = med_mad([r["main_per_step_s"] * 1e6 for r in off])
        m1, m1m = med_mad([r["main_per_step_s"] * 1e6 for r in on])
        med[key] = dict(stage_off=so, stage_on=s1, main_off=mo, main_on=m1, n=min(len(off), len(on)))
        print("%-6s %4d %-11s %-7s | %8.2f±%5.2f %8.2f±%5.2f %7.2f (%5.2f%%) | %8.1f±%5.1f %8.1f±%5.1f %7.1f (%5.2f%%)" % (
            key[0], key[1], key[2], key[3], so, sm, s1, s1m, so - s1, (so - s1) / so * 100,
            mo, mom, m1, m1m, mo - m1, (mo - m1) / mo * 100))

    # rankings
    print("\nRankings on the stage median (ratio = num/den), graph off -> on:")
    flips = []; maxmove = 0.0
    machines = sorted(set(k[0] for k in med))
    for mach in machines:
        for p in sorted(set(k[1] for k in med if k[0] == mach)):
            shapes = sorted(set(k[3] for k in med if k[0] == mach and k[1] == p))
            for shape in shapes:
                def t(path, g):
                    k = (mach, p, path, shape)
                    if k not in med:
                        k = (mach, p, path, "8x8x4")     # non-CUTLASS paths have no shape
                    return med.get(k, {}).get("stage_" + g)
                for num, den in RATIOS:
                    a0, b0, a1, b1 = t(num, "off"), t(den, "off"), t(num, "on"), t(den, "on")
                    if None in (a0, b0, a1, b1):
                        continue
                    r0, r1 = a0 / b0, a1 / b1
                    move = (r1 - r0) * 100
                    maxmove = max(maxmove, abs(move))
                    flag = "FLIP" if (r0 - 1) * (r1 - 1) < 0 else ""
                    if flag:
                        flips.append((mach, p, shape, num, den, r0, r1))
                    print("  %-6s p=%-3d %-7s %-10s/%-10s off %.4f  on %.4f  move %+.2f pt %s" % (
                        mach, p, shape, num, den, r0, r1, move, flag))
                paths = [k[2] for k in med if k[0] == mach and k[1] == p and (k[3] == shape or k[3] == "8x8x4")]
                paths = sorted(set(paths))
                f0 = min(paths, key=lambda q: t(q, "off")); f1 = min(paths, key=lambda q: t(q, "on"))
                print("  %-6s p=%-3d %-7s fastest: off %s  on %s %s" % (
                    mach, p, shape, f0, f1, "" if f0 == f1 else "FLIP"))
                if f0 != f1:
                    flips.append((mach, p, shape, "fastest", "", f0, f1))
    print("\nsign flips: %d   max |move| of a ratio: %.2f pt" % (len(flips), maxmove))
    for fl in flips:
        print("  ", fl)


if __name__ == "__main__":
    main()
