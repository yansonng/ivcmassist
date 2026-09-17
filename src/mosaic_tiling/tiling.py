"""Splitting a large mosaic into model-sized tiles and blending them back.

The three functions here are pure NumPy and carry no dependency on Keras or
Torch, so they can be tested -- and reused with a different segmentation
backend -- without a GPU or a trained model.
"""

from typing import List, Tuple

import numpy as np

from .config import DEFAULT_OVERLAP, DEFAULT_TILE_SIZE


def compute_grid(
    h: int, w: int, tile_h: int, tile_w: int, ov_h: int, ov_w: int
) -> List[Tuple[int, int]]:
    """Return top-left ``(y, x)`` offsets covering an ``h x w`` image.

    Tiles are placed on a regular stride of ``tile - overlap``. Because the
    image size is generally not a multiple of the stride, the last row and
    column are shifted back so that they end exactly at the image border --
    the final tiles therefore overlap their neighbours by more than ``ov_h`` /
    ``ov_w``. The blending weights in :func:`make_weight_window` handle that
    uneven overlap without seams.
    """
    if h < tile_h or w < tile_w:
        raise ValueError(
            f"image {h}x{w} is smaller than the tile {tile_h}x{tile_w}; "
            "pad the image before tiling"
        )

    stride_h = tile_h - ov_h
    stride_w = tile_w - ov_w

    ys = list(range(0, max(h - tile_h, 0) + 1, stride_h))
    xs = list(range(0, max(w - tile_w, 0) + 1, stride_w))

    if ys[-1] != h - tile_h:
        ys.append(h - tile_h)
    if xs[-1] != w - tile_w:
        xs.append(w - tile_w)

    return [(y, x) for y in ys for x in xs]


def crop_with_border(
    img: np.ndarray,
    y: int,
    x: int,
    tile_h: int,
    tile_w: int,
    border_mode: str = "reflect",
    border_constant: int = 0,
):
    """Cut one tile at ``(y, x)``, padding where the tile leaves the image.

    Returns ``(tile, (ystart, yend, xstart, xend))`` where the second element
    marks the region *inside* the returned tile that came from real image
    pixels rather than padding. Only that region is written back during
    stitching, so padded borders never contaminate the output.
    """
    H, W = img.shape[:2]
    y0, x0 = y, x
    y1, x1 = y + tile_h, x + tile_w

    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - H)
    pad_right = max(0, x1 - W)

    y0c = max(0, y0)
    x0c = max(0, x0)
    y1c = min(H, y1)
    x1c = min(W, x1)

    tile = img[y0c:y1c, x0c:x1c]

    if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right)) + ((0, 0),) * (tile.ndim - 2)
        if border_mode == "reflect":
            tile = np.pad(tile, pad_width, mode="reflect")
        else:
            tile = np.pad(tile, pad_width, mode="constant", constant_values=border_constant)

    ystart = pad_top
    xstart = pad_left
    yend = ystart + (y1c - y0c)
    xend = xstart + (x1c - x0c)

    return tile, (ystart, yend, xstart, xend)


def make_weight_window(tile_h: int, tile_w: int, ov_h: int, ov_w: int) -> np.ndarray:
    """Separable raised-cosine window used to blend overlapping predictions.

    The window is ~1 in the tile centre and tapers to 0 at the edges over the
    overlap width, so a pixel predicted by several tiles is dominated by the
    tile that saw it furthest from its own border. Predictions are accumulated
    as a weighted sum and divided by the accumulated weight, which makes the
    blend a proper partition of unity regardless of how tiles overlap.
    """

    def ramp(length: int, overlap: int) -> np.ndarray:
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


def stitch(
    tiles: List[Tuple[np.ndarray, dict]],
    height: int,
    width: int,
    tile_size: Tuple[int, int] = DEFAULT_TILE_SIZE,
    overlap: Tuple[int, int] = DEFAULT_OVERLAP,
) -> np.ndarray:
    """Blend per-tile probability maps back into one ``height x width`` map.

    ``tiles`` is a sequence of ``(prediction, placement)`` pairs, where
    ``placement`` carries the ``y``/``x``/``ystart``/``yend``/``xstart``/``xend``
    keys produced alongside each tile by :func:`crop_with_border`. Predictions
    may be given in ``[0, 1]`` or as ``0-255`` masks; anything with a maximum
    above 1 is rescaled. The result is float in ``[0, 1]``.
    """
    tile_h, tile_w = tile_size
    ov_h, ov_w = overlap
    weight = make_weight_window(tile_h, tile_w, ov_h, ov_w)

    acc = np.zeros((height, width), dtype=np.float32)
    wsum = np.zeros((height, width), dtype=np.float32)

    for pred, rec in tiles:
        if pred.ndim == 3:
            pred = pred[..., 0]
        pred = pred.astype(np.float32)
        if pred.max() > 1.0:
            pred = pred / 255.0

        y, x = rec["y"], rec["x"]
        pred_valid = pred[rec["ystart"]:rec["yend"], rec["xstart"]:rec["xend"]]
        w_valid = weight[rec["ystart"]:rec["yend"], rec["xstart"]:rec["xend"]]

        acc[y:y + pred_valid.shape[0], x:x + pred_valid.shape[1]] += pred_valid * w_valid
        wsum[y:y + pred_valid.shape[0], x:x + pred_valid.shape[1]] += w_valid

    wsum[wsum == 0] = 1.0
    return np.clip(acc / wsum, 0, 1)
