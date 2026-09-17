# Tiled segmentation of corneal nerves and dendritic cells in IVCM mosaics

Segments large in-vivo confocal microscopy (IVCM) mosaics of the corneal
sub-basal nerve plexus. Mosaics are far larger than the 384x384 input a ResUNet
accepts, so this package splits a mosaic into overlapping tiles, segments each
tile, and blends the tile predictions back into one full-resolution mask with a
raised-cosine window that leaves no visible seams.

Two models are applied. The **nerve** model segments nerve fibres; the
**dendritic cell (DC)** model segments dendritic cells, part of which the nerve
model mistakes for nerve. Subtracting the predicted DC pixels from the nerve
mask is what the published nerve measurements were taken from.

This work builds on, and fully covers, the work published in
[Ji, Song et al., *Scientific Reports* 16, 1620 (2026)](https://doi.org/10.1038/s41598-025-34412-6),
which evaluated deep learning segmentation on individual IVCM images. The
tiling and stitching for whole mosaics, and the mosaic nerve models published
here, extend that work and are not evaluated in the article.

The code is deliberately minimal: load a published model, segment one mosaic,
subtract dendritic cells, write the mask. Training, evaluation and the nerve
measurements themselves are not included.

## Install

```bash
git clone https://github.com/yansonng/ivcmassist.git
cd ivcmassist
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e .
```

Tested with Python 3.12.3. `requirements.txt` pins the exact versions the
bit-exact verification ran on; `pip install -e .` instead uses the looser
bounds in `pyproject.toml` if you would rather have a current stack. Keras runs on the PyTorch backend (`KERAS_BACKEND=torch`,
set by the package on import). For GPU inference install the torch build
matching your driver from [pytorch.org](https://pytorch.org/get-started/locally/);
CPU works and is only a few times slower.

## Use

```bash
python scripts/download_weights.py --dest weights/

# 1. segment nerves
python scripts/predict.py --image mosaic.tif \
    --model weights/mosaic_nerves_v2.keras --output nerves.tif

# 2. segment dendritic cells
python scripts/predict.py --image mosaic.tif \
    --model weights/mosaic_dc.keras --output dc.tif

# 3. remove dendritic cells from the nerve mask
python scripts/subtract_dc.py --nerves-mask nerves.tif --dc-mask dc.tif \
    --output nerves_clean.tif
```

`nerves_clean.tif` is the mask the published nerve measurements were taken
from. **Measuring the raw nerve mask instead will not reproduce the study's
numbers** — dendritic cells segmented as nerve inflate the length (2.5% of
nerve pixels on the mosaic used to verify this).

The input is a single-channel `.tif` mosaic of at least 384x384 pixels. The
output is an 8-bit mask at the input resolution.

Mask values depend on the model: the **nerve** models emit 0 and 255
(background, nerve); the **dendritic cell** model emits 0, 127 and 255
(background, type 1, type 2).

Options: `--tile-size H W`, `--overlap H W`, `--border-mode reflect|constant`,
`--device auto|cpu|cuda|cuda:N`, `--keep-tiles DIR` (dump per-tile inputs and
predictions), `--quiet`.

`subtract_dc.py` takes `--threshold` (default 30: the DC mask value above which
a nerve pixel is removed).

Or from Python:

```python
from mosaic_tiling.models import load_segmentation_model
from mosaic_tiling.inference import predict_mosaic
from mosaic_tiling.postprocess import subtract_dc

nerves = predict_mosaic("mosaic.tif", load_segmentation_model(NERVES), "nerves.tif")
dc = predict_mosaic("mosaic.tif", load_segmentation_model(DC), "dc.tif")
clean = subtract_dc(nerves, dc)
```

## Models

Three weight files are published, forming two sets:

| File | Used by | Segments |
|---|---|---|
| `mosaic_nerves_v1.keras` | v1 | corneal nerve fibres |
| `mosaic_nerves_v2.keras` | v2 | corneal nerve fibres (newer; ~25x more training crops) |
| `mosaic_dc.keras` | **v1 and v2** | dendritic cells, types 1 and 2 |

The dendritic cell model is published as part of both sets. It appears once
because the file is the same in each — no DC model was trained after v1 — not
because it is missing from one of them.

See [MODELS.md](MODELS.md) for what each was trained on, its architecture,
licence and checksum.

The models expect mosaics acquired at a 400 um field of view sampled at 384 px
(~1.042 um/px). Other devices or magnifications are untested.

## Known behaviour

Documented deliberately, because each affects the output:

- **Tiles are discretised before blending.** Each tile is reduced to a class
  mask by `argmax`, then rescaled by the largest class index present *in that
  tile* — 0/255 for the 2-class nerve models, 0/127/255 for the 3-class DC
  model. Blending happens on those masks rather than on raw probabilities.
  This is what produced the reference study masks, so it is preserved here;
  blending probabilities would be the more natural choice for new work.
- **Per-tile rescaling has an edge case for dendritic cells.** In a tile
  containing type 1 cells and no type 2, the largest index present is 1, so
  type 1 pixels are written as 255 — the value that means type 2 elsewhere.
  Overlapping neighbours normally outvote it: on the mosaics checked, one such
  tile occurred in 48 and the type 1 / type 2 counts were unchanged. A mosaic
  small enough to be covered by one or two tiles has no neighbour to correct
  it.
- **The outer 1-pixel frame of every stitched mask is zero.** The blending
  window is exactly 0 along a tile's edge, and the mosaic border is covered
  only by tiles whose edge lands there, so the accumulated weight is 0. On a
  ~2000x2000 mosaic this is 0.26% of pixels; it is asserted in the test suite.
- **Mosaics smaller than one tile are rejected** with a clear error. The
  original code silently produced a negative tile offset in that case and
  stitched the tile into the wrong location.

## Reproducing the study masks

The tiling code is numerically identical to the study code. `tests/` contains
the original implementations verbatim and asserts equality against them, and
`scripts/regression_check.py` compares whole stitched masks:

```bash
pip install -r requirements-dev.txt
pytest                                    # 42 equivalence and property tests

python scripts/regression_check.py \
    --predicted-dir mask.tif --reference-dir reference_mask.tif
```

Against masks produced by the original pipeline, this package reproduces every
mask bit-for-bit — verified for all three published models and for the
DC-subtracted nerve masks of both sets, on mosaics from 1 to 210 tiles.

**Bit-exact reproduction is GPU-specific.** The reference masks were produced
on CUDA; running the same model on CPU flips `argmax` at a small number of
borderline pixels — measured at 58 of 3,798,652 pixels (0.0015%), a 0.02%
difference in segmented foreground. That is ordinary floating-point
non-determinism between backends, not a difference in the method. Compare with
`--tolerance` or expect exactness only on the same hardware class.

## Data

No patient imaging data is included. Corneal IVCM mosaics are identifiable
clinical data; see the data availability statement in the paper.

## Citing

If you use this code or the models, please cite:

> Ji, M., Song, Y., Roth, J., Dashti, A., Lazo, J., Lincke, A., Macedo, A. F. T.,
> Löwe, W. & Lagali, N. Deep learning-based segmentation and density estimation
> of corneal nerves and dendritic cells from In Vivo confocal microscopy images.
> *Scientific Reports* **16**, 1620 (2026).
> https://doi.org/10.1038/s41598-025-34412-6

Machine-readable metadata, including the software record, is in
[CITATION.cff](CITATION.cff).

## Licence

Code is Apache-2.0 ([LICENSE](LICENSE)). Model weights and documentation are
CC-BY-4.0; see [MODELS.md](MODELS.md).
