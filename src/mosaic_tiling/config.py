"""Tiling defaults.

The published models were trained on 384x384 tiles from mosaics acquired at a
400 um field of view sampled at 384 px (~1.042 um/px). Applying them to images
from another device or magnification is untested.
"""

from typing import Tuple

#: Input size the ResUNet models were trained on (height, width), in pixels.
DEFAULT_TILE_SIZE: Tuple[int, int] = (384, 384)

#: Overlap between neighbouring tiles (height, width), in pixels.
DEFAULT_OVERLAP: Tuple[int, int] = (64, 64)
