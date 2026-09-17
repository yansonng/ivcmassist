"""Tiled segmentation of large IVCM corneal mosaics.

Splits a mosaic into overlapping model-sized tiles, segments each tile with a
ResUNet, and blends the tiles back into one full-resolution mask.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # must precede any keras import

from .config import DEFAULT_OVERLAP, DEFAULT_TILE_SIZE  # noqa: E402
from .tiling import compute_grid, crop_with_border, make_weight_window, stitch  # noqa: E402

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_OVERLAP",
    "DEFAULT_TILE_SIZE",
    "compute_grid",
    "crop_with_border",
    "make_weight_window",
    "stitch",
    "__version__",
]
