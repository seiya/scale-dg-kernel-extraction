#!/usr/bin/env python3
"""Estimate and calibrate device memory for the high-order GEMM paths."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


GIB = 1 << 30
MEMORY_RE = re.compile(
    r"^DEVICE_MEMORY\s+(\S+)\s+total=(\d+)\s+used=(\d+)\s+free=(\d+)\s*$"
)
CONFIG_RE = re.compile(
    r"^DEVICE_MEMORY_CONFIG\s+poly_order=(\d+)\s+np=(\d+)\s+ne=(\d+)\s+nea=(\d+)\s*$"
)


def allocation_bytes(poly_order: int, nex: int, ney: int, nez: int) -> dict[str, int]:
    nq = poly_order + 1
    np = nq**3
    nfp = nq**2
    nfptot = 6 * nfp
    ne = nex * ney * nez
    nhalo = 2 * (ney * nez + nex * nez + nex * ney)
    nhalo_node = nfp * nhalo
    nea = ne + (nhalo_node + np - 1) // np

    # Both current GEMM paths have the same storage: z is written directly to
    # dqdt and surface_lift is not allocated. Default INTEGER is four bytes
    # and RP is FP64 in the benchmark builds covered by this estimator.
    return {
        "q/u/v/w (packed halo)": 4 * np * nea * 8,
        "q0/dqdt (owned)": 2 * np * ne * 8,
        "D1D/D1D_tr": 2 * nq * nq * 8,
        "Lift1D": 6 * nq * 8,
        "Lift_mat (dummy for Nq>128)": (8 if nq > 128 else 6 * np * 8),
        "VMapM/VMapP": 2 * nfptot * ne * 4,
        "normal_fn": 3 * nfptot * ne * 8,
        "Escale": 3 * np * ne * 8,
        "Fscale": nfptot * ne * 8,
        "halo_src_map": nhalo_node * 4,
        "ebnd_flux": nfptot * ne * 8,
        "volume_flux_x/y/z": 3 * np * ne * 8,
        "volume_deriv_x/y": 2 * np * ne * 8,
    }


def parse_memory_log(path: Path) -> tuple[list[tuple[str, int, int, int]], tuple[int, ...] | None]:
    samples = []
    config = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = MEMORY_RE.match(line.strip())
        if match:
            label, total, used, free = match.groups()
            samples.append((label, int(total), int(used), int(free)))
        match = CONFIG_RE.match(line.strip())
        if match:
            config = tuple(int(value) for value in match.groups())
    if not samples:
        raise SystemExit(f"no DEVICE_MEMORY samples found in {path}")
    return samples, config


def gib(value: int) -> str:
    return f"{value / GIB:.3f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact application-array bytes for CUDAFORTRAN_GEMM and "
            "CUDAFORTRAN_GEMM_FUSED, optionally calibrated by runtime output."
        )
    )
    parser.add_argument("--poly-order", type=int, required=True)
    parser.add_argument("--nex", type=int, default=1)
    parser.add_argument("--ney", type=int, default=1)
    parser.add_argument("--nez", type=int, default=1)
    parser.add_argument(
        "--path", choices=("GEMM", "GEMM_FUSED"), default="GEMM_FUSED"
    )
    parser.add_argument(
        "--memory-log",
        type=Path,
        help="program output produced with SCALE_DG_REPORT_DEVICE_MEMORY=1",
    )
    parser.add_argument(
        "--safety-gib",
        type=float,
        default=2.0,
        help="extra free-memory margin for the feasibility check (default: 2)",
    )
    parser.add_argument(
        "--allocator-overhead",
        type=float,
        default=0.25,
        help=(
            "conservative allocator allowance as a fraction of payload "
            "(default: 0.25, bounding p=511/575/767/1023 measurements)"
        ),
    )
    args = parser.parse_args()

    if min(args.poly_order, args.nex, args.ney, args.nez) < 1:
        parser.error("poly order and element counts must be positive")

    components = allocation_bytes(args.poly_order, args.nex, args.ney, args.nez)
    requested = sum(components.values())
    safety = round(args.safety_gib * GIB)
    predicted = round(requested * (1.0 + args.allocator_overhead))

    print(
        f"p={args.poly_order}, elements={args.nex}x{args.ney}x{args.nez}, "
        f"path=CUDAFORTRAN_{args.path}"
    )
    print("\nApplication-requested device payload:")
    for name, value in components.items():
        print(f"  {name:34s} {value:16d}  {gib(value):>12s}")
    print(f"  {'TOTAL':34s} {requested:16d}  {gib(requested):>12s}")
    print(
        f"\nConservative estimate ({args.allocator_overhead:.0%} allocator allowance):"
    )
    print(f"  payload plus allocator allowance      {gib(predicted):>12s}")
    print(f"  plus safety margin                    {gib(predicted + safety):>12s}")

    if args.memory_log is None:
        print(
            "\nThis is array payload only. Run with "
            "SCALE_DG_REPORT_DEVICE_MEMORY=1 and pass --memory-log to include "
            "OpenACC/CUDA/library reserve."
        )
        return

    samples, config = parse_memory_log(args.memory_log)
    expected_ne = args.nex * args.ney * args.nez
    if config is not None and (config[0] != args.poly_order or config[2] != expected_ne):
        raise SystemExit(
            "memory-log configuration does not match --poly-order/element counts: "
            f"log has p={config[0]}, Ne={config[2]}"
        )
    startup = next((sample for sample in samples if sample[0] == "startup"), None)
    if startup is None:
        raise SystemExit("memory log has no startup sample")
    peak = max(samples, key=lambda sample: sample[2])
    if any(sample[1] != startup[1] for sample in samples):
        raise SystemExit("device total changed between memory samples")

    actual_increment = peak[2] - startup[2]
    reserve = actual_increment - requested
    calibrated_need = actual_increment + safety

    print("\nCUDA-visible samples:")
    for label, total, used, free in samples:
        print(f"  {label:24s} used {gib(used):>12s}  free {gib(free):>12s}")
    print(f"\n  startup free                         {gib(startup[3]):>12s}")
    print(f"  peak incremental use ({peak[0]}) {gib(actual_increment):>12s}")
    print(f"  requested array payload              {gib(requested):>12s}")
    print(f"  measured runtime/allocator delta     {gib(reserve):>12s}")
    print(f"  configured safety margin             {gib(safety):>12s}")
    print(f"  calibrated need incl. safety         {gib(calibrated_need):>12s}")
    print(
        "  feasible at measured startup free    "
        + ("yes" if calibrated_need <= startup[3] else "NO")
    )
    print(
        "  conservative estimate incl. safety   "
        + ("yes" if predicted + safety <= startup[3] else "NO")
        + "  (feasible at startup free)"
    )
    print(
        "\nThe runtime/allocator delta is empirical and must be remeasured after "
        "changing the compiler, driver, allocator mode, or allocation sequence."
    )


if __name__ == "__main__":
    main()
