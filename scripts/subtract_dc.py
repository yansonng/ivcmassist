#!/usr/bin/env python3
"""Remove dendritic cell pixels from a nerve mask.

Third step of the study pipeline, after segmenting a mosaic with the nerve
model and with the dendritic cell model:

    python scripts/subtract_dc.py \
        --nerves-mask nerves.tif --dc-mask dc.tif --output nerves_clean.tif

The nerve model segments part of each dendritic cell as nerve. Nerve
measurements in the study pipeline were taken from the cleaned mask this produces, not
from the raw nerve mask.
"""

import argparse
import sys
from pathlib import Path

import cv2
from tifffile import imwrite

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mosaic_tiling.postprocess import DC_MASK_THRESHOLD, subtract_dc


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--nerves-mask", required=True, type=Path, help="mask from a nerve model")
    p.add_argument("--dc-mask", required=True, type=Path, help="mask from the DC model")
    p.add_argument("--output", required=True, type=Path, help="where to write the cleaned mask")
    p.add_argument("--threshold", type=int, default=DC_MASK_THRESHOLD,
                   help=f"DC mask value above which nerve pixels are removed "
                        f"(default: {DC_MASK_THRESHOLD})")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    for path in (args.nerves_mask, args.dc_mask):
        if not path.is_file():
            raise SystemExit(f"mask not found: {path}")

    nerves = cv2.imread(str(args.nerves_mask), cv2.IMREAD_GRAYSCALE)
    dc = cv2.imread(str(args.dc_mask), cv2.IMREAD_GRAYSCALE)
    if nerves is None or dc is None:
        raise SystemExit("could not read one of the masks as 8-bit grayscale")

    cleaned = subtract_dc(nerves, dc, args.threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(args.output), cleaned, compression=None)

    if not args.quiet:
        removed = int((nerves > 127).sum() - (cleaned > 127).sum())
        total = int((nerves > 127).sum())
        pct = 100 * removed / total if total else 0.0
        print(f"Removed {removed} of {total} nerve pixels ({pct:.2f}%) as dendritic cell")
        print(f"Saved cleaned mask: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
