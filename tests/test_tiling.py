"""Equivalence tests against the original study implementation.

The reference functions below are verbatim copies of the tiling code used to
produce the study's reference masks. Every test asserts that the packaged version is
numerically identical to them, so a refactor can never silently change what the
paper reported.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mosaic_tiling.tiling import (  # noqa: E402
    compute_grid,
    crop_with_border,
    make_weight_window,
    stitch,
)

BORDER_MODE = "reflect"
BORDER_CONSTANT = 0


# --------------------------------------------------------------------------
# Reference implementation (original study code, unmodified)
# --------------------------------------------------------------------------
def ref_compute_grid(h, w, tile_h, tile_w, ov_h, ov_w):
    stride_h = tile_h - ov_h
    stride_w = tile_w - ov_w
    ys = list(range(0, max(h - tile_h, 0) + 1, stride_h))
    xs = list(range(0, max(w - tile_w, 0) + 1, stride_w))
    if ys[-1] != h - tile_h:
        ys.append(h - tile_h)
    if xs[-1] != w - tile_w:
        xs.append(w - tile_w)
    return [(y, x) for y in ys for x in xs]


def ref_crop_with_border(img, y, x, tile_h, tile_w):
    H, W = img.shape[:2]
    y0, x0 = y, x
    y1, x1 = y + tile_h, x + tile_w
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - H)
    pad_right = max(0, x1 - W)
    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(H, y1), min(W, x1)
    tile = img[y0c:y1c, x0c:x1c]
    if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
        pad = ((pad_top, pad_bottom), (pad_left, pad_right)) + ((0, 0),) * (tile.ndim - 2)
        if BORDER_MODE == "reflect":
            tile = np.pad(tile, pad, mode="reflect")
        else:
            tile = np.pad(tile, pad, mode="constant", constant_values=BORDER_CONSTANT)
    ystart, xstart = pad_top, pad_left
    yend = ystart + (y1c - y0c)
    xend = xstart + (x1c - x0c)
    return tile, (ystart, yend, xstart, xend)


def ref_make_weight_window(tile_h, tile_w, ov_h, ov_w):
    def ramp(length, overlap):
        if overlap <= 0:
            return np.ones(length, dtype=np.float32)
        x = np.linspace(0, 1, length, dtype=np.float32)
        d = np.minimum(x, 1 - x)
        eps = 1e-6
        r = np.clip(d / (overlap / (length + eps)), 0, 1)
        return (0.5 - 0.5 * np.cos(np.pi * r)).astype(np.float32)

    wy = ramp(tile_h, ov_h)
    wx = ramp(tile_w, ov_w)
    w2d = np.outer(wy, wx)
    m = w2d.max()
    if m > 0:
        w2d = w2d / m
    return w2d.astype(np.float32)


# Mosaic sizes spanning exact multiples of the stride and awkward remainders.
SHAPES = [(384, 384), (700, 512), (1024, 1024), (1876, 1280), (2000, 3011), (1152, 897)]
TILINGS = [((384, 384), (64, 64)), ((256, 256), (32, 32)), ((384, 384), (0, 0))]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tile_size,overlap", TILINGS)
def test_grid_matches_reference(shape, tile_size, overlap):
    h, w = shape
    assert compute_grid(h, w, *tile_size, *overlap) == ref_compute_grid(
        h, w, *tile_size, *overlap
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_grid_covers_every_pixel(shape):
    h, w = shape
    tile_h, tile_w = 384, 384
    covered = np.zeros((h, w), dtype=bool)
    for y, x in compute_grid(h, w, tile_h, tile_w, 64, 64):
        covered[y:y + tile_h, x:x + tile_w] = True
    assert covered.all(), "tile grid leaves part of the mosaic unsegmented"


def test_grid_rejects_undersized_image():
    # The original returned a negative offset here, which silently wrote the
    # tile to the wrong place during stitching.
    with pytest.raises(ValueError, match="smaller than the tile"):
        compute_grid(200, 800, 384, 384, 64, 64)


@pytest.mark.parametrize("shape", SHAPES)
def test_crop_matches_reference(shape):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=shape, dtype=np.uint8)
    for y, x in compute_grid(*shape, 384, 384, 64, 64):
        tile, region = crop_with_border(img, y, x, 384, 384)
        ref_tile, ref_region = ref_crop_with_border(img, y, x, 384, 384)
        assert np.array_equal(tile, ref_tile)
        assert region == ref_region
        assert tile.shape == (384, 384)


@pytest.mark.parametrize("tile_size,overlap", TILINGS)
def test_weight_window_matches_reference(tile_size, overlap):
    assert np.array_equal(
        make_weight_window(*tile_size, *overlap),
        ref_make_weight_window(*tile_size, *overlap),
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_stitch_is_a_partition_of_unity_in_the_interior(shape):
    """Constant in, constant out everywhere except the outer 1-pixel frame.

    The raised-cosine window is exactly 0 along a tile's own edge. Interior
    pixels are always covered by a neighbouring tile that weights them above
    zero, but the mosaic's outermost row and column are covered *only* by tiles
    whose edge lands there, so their accumulated weight is 0 and stitching
    emits 0. See "Known behaviour" in the README.
    """
    h, w = shape
    tiles = []
    for y, x in compute_grid(h, w, 384, 384, 64, 64):
        _, region = crop_with_border(np.zeros(shape, np.uint8), y, x, 384, 384)
        ystart, yend, xstart, xend = region
        tiles.append(
            (
                np.full((384, 384), 255, np.uint8),
                {"y": y, "x": x, "ystart": ystart, "yend": yend, "xstart": xstart, "xend": xend},
            )
        )
    out = stitch(tiles, h, w)

    assert np.allclose(out[1:-1, 1:-1], 1.0, atol=1e-6)

    deviating = np.argwhere(~np.isclose(out, 1.0, atol=1e-6))
    assert len(deviating) == 2 * (h + w) - 4, "deviation is wider than a 1-pixel frame"
    assert (out[deviating[:, 0], deviating[:, 1]] == 0).all()


@pytest.mark.parametrize("shape", [(700, 512), (1876, 1280)])
def test_stitch_roundtrip_recovers_structure(shape):
    """A mask tiled and stitched back reproduces the original, frame aside."""
    h, w = shape
    rng = np.random.default_rng(42)
    truth = (rng.random(shape) > 0.7).astype(np.uint8) * 255

    tiles = []
    for y, x in compute_grid(h, w, 384, 384, 64, 64):
        tile, region = crop_with_border(truth, y, x, 384, 384)
        ystart, yend, xstart, xend = region
        tiles.append(
            (tile, {"y": y, "x": x, "ystart": ystart, "yend": yend, "xstart": xstart, "xend": xend})
        )

    out = (stitch(tiles, h, w) * 255).astype(np.uint8)
    assert np.array_equal(out[1:-1, 1:-1], truth[1:-1, 1:-1])
