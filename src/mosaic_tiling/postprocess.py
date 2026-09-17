"""Removing dendritic cells from a nerve mask.

The nerve model segments part of each dendritic cell as nerve, which inflates
measured nerve length. The study pipeline therefore ran both models over a
mosaic and subtracted the predicted dendritic cell pixels from the nerve mask
before measuring. Masks produced without this step do not correspond to the
study's measurements.
"""

import numpy as np

#: Pixels above this value in the DC mask are treated as dendritic cell.
#: Type 2 cells are white (255) and type 1 grey (127), so a low threshold
#: removes both; 30 rather than 0 leaves a margin for blending artefacts.
DC_MASK_THRESHOLD = 30


def subtract_dc(
    nerves_mask: np.ndarray,
    dc_mask: np.ndarray,
    threshold: int = DC_MASK_THRESHOLD,
) -> np.ndarray:
    """Zero out nerve pixels wherever the dendritic cell mask fires."""
    if nerves_mask.shape != dc_mask.shape:
        raise ValueError(
            f"mask shapes differ: nerves {nerves_mask.shape} vs DC {dc_mask.shape}; "
            "both must come from the same mosaic"
        )
    return np.where(dc_mask > threshold, 0, nerves_mask).astype(np.uint8)
