#!/usr/bin/env python3
"""Segment one IVCM mosaic with a published model.

    python scripts/predict.py --image mosaic.tif \
        --model weights/mosaic_nerves_v2.keras --output mask.tif

The mosaic is split into overlapping 384x384 tiles, each tile is segmented, and
the tile predictions are blended back into one full-resolution mask.

Mask values depend on the model: the nerve models emit 0 and 255 (background,
nerve); the dendritic cell model emits 0, 127 and 255 (background, type 1,
type 2).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mosaic_tiling.config import DEFAULT_OVERLAP, DEFAULT_TILE_SIZE
from mosaic_tiling.inference import predict_mosaic, resolve_device
from mosaic_tiling.models import load_segmentation_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image", required=True, type=Path, help="mosaic .tif to segment")
    p.add_argument("--model", required=True, type=Path, help="published .keras model")
    p.add_argument("--output", required=True, type=Path, help="where to write the mask")
    p.add_argument("--tile-size", type=int, nargs=2, default=list(DEFAULT_TILE_SIZE),
                   metavar=("H", "W"), help="tile size in pixels (default: 384 384)")
    p.add_argument("--overlap", type=int, nargs=2, default=list(DEFAULT_OVERLAP),
                   metavar=("H", "W"), help="tile overlap in pixels (default: 64 64)")
    p.add_argument("--border-mode", choices=("reflect", "constant"), default="reflect",
                   help="padding where a tile leaves the mosaic (default: reflect)")
    p.add_argument("--border-constant", type=int, default=0,
                   help="fill value when --border-mode constant")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N (default: auto)")
    p.add_argument("--keep-tiles", type=Path, default=None,
                   help="directory to dump per-tile inputs and predictions for inspection")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.image.is_file():
        raise SystemExit(f"image not found: {args.image}")
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")

    device = resolve_device(args.device)
    if not args.quiet:
        print(f"Device: {device}")
        print(f"Model:  {args.model}")

    model = load_segmentation_model(args.model)
    predict_mosaic(
        tif_path=args.image,
        model=model,
        output_mask_path=args.output,
        tile_size=tuple(args.tile_size),
        overlap=tuple(args.overlap),
        border_mode=args.border_mode,
        border_constant=args.border_constant,
        tiles_dir=args.keep_tiles,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
