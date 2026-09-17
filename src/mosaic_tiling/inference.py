"""Tile a mosaic, segment each tile, blend the tiles back into one mask.

The numerical path deliberately matches the original study code: each tile is
segmented, reduced to a discrete mask with ``argmax`` over the output channels,
and only then blended. Blending discretised tiles rather than raw probabilities
is what the reference study masks were produced with, so it is kept here even
though blending probabilities would be the more obvious choice.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from tifffile import imread, imwrite

from .config import DEFAULT_OVERLAP, DEFAULT_TILE_SIZE
from .tiling import compute_grid, crop_with_border, stitch


def resolve_device(device: str = "auto") -> torch.device:
    """Turn ``auto`` / ``cpu`` / ``cuda`` / ``cuda:N`` into a torch device."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available; use --device cpu")
        if resolved.index is not None:
            if resolved.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {resolved.index} requested but only "
                    f"{torch.cuda.device_count()} device(s) present"
                )
            torch.cuda.set_device(resolved.index)
    return resolved


def normalize_tile(tile: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """Min-max normalise a tile to [0, 1] and shape it as ``(1, H, W, 1)``."""
    image = tile.astype("float32")
    span = image.max() - image.min()
    if span == 0:
        image = np.zeros_like(image, dtype=np.float32)
    else:
        image = (image - image.min()) / span

    if image.shape[:2] != input_size:
        image = cv2.resize(image, (input_size[1], input_size[0]))

    return image[np.newaxis, ..., np.newaxis]


def predict_tile(model, tile: np.ndarray, input_size: Tuple[int, int] = DEFAULT_TILE_SIZE) -> np.ndarray:
    """Segment one tile, returning a ``uint8`` class mask scaled to 0-255.

    The winning class index is rescaled by the largest index *present in this
    tile*, which spreads the classes over 0-255: a 2-class nerve tile becomes
    0/255, and a 3-class dendritic cell tile becomes 0/127/255.

    That per-tile rescaling is how the reference study masks were produced and is
    kept for reproducibility, but note the edge case it carries: in a tile
    where the model predicts type 1 cells and no type 2, the largest index
    present is 1, so type 1 pixels are written as 255 -- the value that means
    type 2 everywhere else. Overlapping neighbours normally outvote such a tile
    during blending (measured: no change to dendritic cell counts on the images
    checked), but a mosaic small enough to be covered by one or two tiles has
    no neighbour to correct it. See "Known behaviour" in the README.
    """
    batch = normalize_tile(tile, input_size)

    with torch.no_grad():
        pred = model(batch)

    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()[0]
    else:
        pred = pred.numpy()[0]

    mask = np.argmax(pred, axis=-1)
    max_val = mask.max()
    if max_val > 0:
        mask = (mask / max_val) * 255
    else:
        mask = np.zeros_like(mask)

    return mask.astype(np.uint8)


def predict_mosaic(
    tif_path,
    model,
    output_mask_path,
    tile_size: Tuple[int, int] = DEFAULT_TILE_SIZE,
    overlap: Tuple[int, int] = DEFAULT_OVERLAP,
    border_mode: str = "reflect",
    border_constant: int = 0,
    tiles_dir: Optional[Path] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Segment one large mosaic TIFF and write the stitched mask.

    Set ``tiles_dir`` to also dump the individual tiles and their predictions,
    which is useful for figures and for debugging a single bad region.
    """
    tif_path = Path(tif_path)
    img = imread(str(tif_path))

    if img.ndim != 2:
        raise ValueError(
            f"{tif_path.name}: expected a single-channel mosaic, got shape {img.shape}. "
            "Convert to grayscale before running the models -- they were trained on "
            "single-channel IVCM images."
        )

    H, W = img.shape
    tile_h, tile_w = tile_size
    coords = compute_grid(H, W, tile_h, tile_w, *overlap)

    if tiles_dir is not None:
        tiles_dir = Path(tiles_dir)
        (tiles_dir / "tiles").mkdir(parents=True, exist_ok=True)
        (tiles_dir / "preds").mkdir(parents=True, exist_ok=True)

    predictions = []
    for i, (y, x) in enumerate(coords, start=1):
        tile, (ystart, yend, xstart, xend) = crop_with_border(
            img, y, x, tile_h, tile_w, border_mode, border_constant
        )
        mask = predict_tile(model, tile, tile_size)

        if tiles_dir is not None:
            name = f"{tif_path.stem}_tile_y{y}_x{x}.tif"
            imwrite(str(tiles_dir / "tiles" / name), tile, compression=None)
            imwrite(str(tiles_dir / "preds" / name), mask, compression=None)

        predictions.append(
            (mask, {"y": y, "x": x, "ystart": ystart, "yend": yend, "xstart": xstart, "xend": xend})
        )

        if verbose and i % 50 == 0:
            print(f"  {tif_path.name}: {i}/{len(coords)} tiles")

    stitched = stitch(predictions, H, W, tile_size, overlap)
    out = (stitched * 255).astype(np.uint8)

    output_mask_path = Path(output_mask_path)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(output_mask_path), out, compression=None)

    if verbose:
        print(f"Saved stitched mask: {output_mask_path} ({len(coords)} tiles)")

    return out
