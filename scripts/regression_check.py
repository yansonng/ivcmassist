#!/usr/bin/env python3
"""Compare masks produced by this package against reference masks.

Use this to prove that the released code reproduces the masks the published
results were computed from: point ``--reference-dir`` at masks produced by the
original study code and ``--predicted-dir`` at masks produced here.

    python scripts/regression_check.py --predicted-dir output/dc \
        --reference-dir /path/to/original/dc_masks
"""

import argparse
from pathlib import Path

import numpy as np
from tifffile import imread


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predicted-dir", required=True, type=Path,
                   help="mask directory, or a single mask file")
    p.add_argument("--reference-dir", required=True, type=Path,
                   help="reference directory, or a single reference file")
    p.add_argument("--tolerance", type=int, default=0,
                   help="max absolute per-pixel difference to accept (default: 0, exact)")
    p.add_argument("--ignore-border", type=int, default=0,
                   help="ignore an N-pixel frame around the mask (default: 0)")
    args = p.parse_args()

    if args.predicted_dir.is_file():
        predicted = [args.predicted_dir]
        reference_of = lambda _p: args.reference_dir
    else:
        predicted = sorted(args.predicted_dir.glob("*.tif"))
        reference_of = lambda p: args.reference_dir / p.name
        if not predicted:
            raise SystemExit(f"no masks in {args.predicted_dir}")

    failures = 0
    compared = 0

    for pred_path in predicted:
        ref_path = reference_of(pred_path)
        if not ref_path.exists():
            print(f"  {pred_path.name}: no reference, skipped")
            continue

        a = imread(str(pred_path)).astype(np.int32)
        b = imread(str(ref_path)).astype(np.int32)

        if a.shape != b.shape:
            print(f"  {pred_path.name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            failures += 1
            continue

        n = args.ignore_border
        if n:
            a, b = a[n:-n, n:-n], b[n:-n, n:-n]

        diff = np.abs(a - b)
        differing = int((diff > args.tolerance).sum())
        compared += 1

        if differing:
            failures += 1
            pct = 100 * differing / diff.size
            print(f"  {pred_path.name}: {differing} px differ ({pct:.4f}%), "
                  f"max diff {diff.max()}")
        else:
            print(f"  {pred_path.name}: identical")

    print(f"\n{compared - failures}/{compared} masks match "
          f"(tolerance {args.tolerance}, border ignored {args.ignore_border})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
