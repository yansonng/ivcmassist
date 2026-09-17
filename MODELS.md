# Published models

Three weight files are published: two nerve models and one dendritic cell
model. They form two sets, v1 and v2, which differ **only in the nerve model**.

The dendritic cell model belongs to both sets. No DC model was trained after
the v1 one, so `mosaic_dc.keras` is byte-identical for v1 and v2 and is
distributed once rather than duplicated — using it with either nerve model is
the intended configuration.

| Set | Nerve model | DC model | Trained on |
|---|---|---|---|
| **v1** | `mosaic_nerves_v1.keras` | `mosaic_dc.keras` | 1,213 IVCM images + 637 mosaic crops |
| **v2** | `mosaic_nerves_v2.keras` | `mosaic_dc.keras` | 16,313 mosaic crops |

v2 is the newer model, trained on roughly 25x more mosaic crops. v1 is the set
behind the earlier study outputs. Both are published so results computed with
either can be reproduced.

## Files

| File | Size | SHA-256 |
|---|---|---|
| `mosaic_nerves_v1.keras` | 133.0 MB | `f9e6bdca92b877e92f4ed8b609ae0ec36519e04f93744c84d5dabf535e33e197` |
| `mosaic_nerves_v2.keras` | 133.0 MB | `2f4b2f9ec62fd0f37567505021d87db435af02ae91e59b56285f4e7fb6badde1` |
| `mosaic_dc.keras` | 133.3 MB | `d91a188d3c69094f9686221f52685fc643ac803073480da2d9f5c887fc43b58c` |

Distributed on HuggingFace at
[`yansonng/ivcmassist`](https://huggingface.co/yansonng/ivcmassist).

Fetch and verify with `python scripts/download_weights.py --dest weights/`.

## Architectures

The nerve and dendritic cell models are **not the same network**:

| Model | Input | Encoder levels | Base filters | Output classes | Mask values | Parameters |
|---|---|---|---|---|---|---|
| nerves (v1, v2) | 384x384x1 | 4 | 64 | 2 (background, nerve) | 0, 255 | 33,156,994 |
| dendritic cells | 384x384x1 | 5 | 32 | 3 (background, type 1, type 2) | 0, 127, 255 | 33,227,235 |

Both are residual U-Nets: each block is two 3x3 convolutions with batch
normalisation, a 1x1 convolution on the skip path, an add and a ReLU, with a
1024-filter bridge and a softmax output.

## Relationship to the article

These models build on
[Ji, Song et al., *Scientific Reports* 16, 1620 (2026)](https://doi.org/10.1038/s41598-025-34412-6),
which evaluated segmentation of individual IVCM images. Read the dates before
attributing any result in the article to a file here:

- The article was published on 2026-01-13. **Both mosaic nerve models were
  trained afterwards** (v1 on 2026-02-15, v2 on 2026-05-18), so neither is the
  nerve model evaluated in the article, and the article reports no results on
  mosaics.
- The dendritic cell model predates the article (trained 2025-04-17) and its
  cross-validation used 1,257 annotated images, close to the 1,300 DC
  annotations the article describes. It has not been confirmed to be the exact
  checkpoint behind the article's DC results.

## Provenance

The published files are the checkpoints that produced the study's mask outputs,
stripped of optimizer state:

| Published as | Internal checkpoint |
|---|---|
| `mosaic_nerves_v1.keras` | `nerves_output/cross_validation/20260215_072712/model_20260215_072712_fold_2.keras` |
| `mosaic_nerves_v2.keras` | `nerves_output/cross_validation/20260518_011017/mosaic_resunet_fold1_20260518_011017.keras` |
| `mosaic_dc.keras` | `dc_output/cross_validation/20250417_033740/model_20250417_033740_fold_4.keras` |

Each was confirmed by re-running it and comparing whole stitched masks against
the stored study outputs — bit-identical on every image tested. For the v1
nerve model the identification is discriminating, not merely consistent: a
different fold of the same cross-validation run differs on ~1% of pixels.

Both nerve models are **cross-validation folds**, not models retrained on all
data.

## Training data

<!-- Fill in from the study records: cohort size, number of mosaics, annotation
     procedure and the split, without patient identifiers. -->

- `mosaic_nerves_v1.keras` — 1,213 IVCM images plus 637 mosaic crops. TODO:
  cohort description and annotation procedure.
- `mosaic_nerves_v2.keras` — 16,313 mosaic crops. TODO: as above.
- `mosaic_dc.keras` — TODO.

## Intended use and limitations

- Trained on IVCM mosaics of the corneal sub-basal nerve plexus acquired at a
  400 um field of view sampled at 384 px. Performance on other devices,
  magnifications or corneal layers is unvalidated.
- **Research use only.** Not a medical device and not validated for clinical
  decision-making.
- The nerve models segment some dendritic cells as nerve. Run the DC model and
  subtract it (`scripts/subtract_dc.py`) before measuring, as the study
  pipeline did — measurements taken from a raw nerve mask will not match the
  study's measurements. The measurement code itself is not part of this release.

## Preparation for release

Published weights are stripped of optimizer state with
`scripts/strip_optimizer.py`, which reduces each file from ~398 MB to ~133 MB
and verifies the stripped model predicts bit-identically before writing it.
Stripping also removes the stored compile configuration, which in the original
checkpoints referenced the training modules by name (`dc_utils.DiceLoss`) and
prevented loading them outside the original working directory.

## Licence

Model weights are released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Attribute by citing
the article; see `CITATION.cff`.
