#!/usr/bin/env python3
"""Remove optimizer state from a .keras checkpoint before publishing it.

Training checkpoints store the Adam moment estimates alongside the weights,
which roughly triples the file size and is useless for inference. Reloading the
model without compiling it and saving it again drops that state.

    python scripts/strip_optimizer.py model.keras -o weights/mosaic_nerves.keras

The stripped file is verified by loading it back and checking it predicts
bit-identically to the original on random inputs, so publishing it cannot
change any reported result.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mosaic_tiling.models import load_segmentation_model  # noqa: E402


def archive_breakdown(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            print(f"    {info.file_size / 1e6:8.1f} MB  {info.filename}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path, help="input .keras file")
    p.add_argument("-o", "--output", type=Path, required=True, help="output .keras file")
    p.add_argument("--input-size", type=int, nargs=2, default=[384, 384], metavar=("H", "W"))
    p.add_argument("--trials", type=int, default=3, help="random inputs used to verify (default: 3)")
    p.add_argument("--skip-verify", action="store_true")
    args = p.parse_args()

    if not args.model.exists():
        raise SystemExit(f"not found: {args.model}")

    before = args.model.stat().st_size
    print(f"Input:  {args.model.name}  {before / 1e6:.1f} MB")
    archive_breakdown(args.model)

    original = load_segmentation_model(args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    original.save(args.output)

    after = args.output.stat().st_size
    print(f"\nOutput: {args.output.name}  {after / 1e6:.1f} MB "
          f"({before / max(after, 1):.1f}x smaller)")
    archive_breakdown(args.output)

    if args.skip_verify:
        return 0

    print("\nVerifying the stripped model predicts identically...")
    stripped = load_segmentation_model(args.output)

    rng = np.random.default_rng(0)
    for trial in range(args.trials):
        batch = rng.random((1, *args.input_size, 1)).astype("float32")
        with torch.no_grad():
            a = np.asarray(original(batch).detach().cpu())
            b = np.asarray(stripped(batch).detach().cpu())
        identical = np.array_equal(a, b)
        print(f"  trial {trial + 1}: max abs diff {np.abs(a - b).max():.3e} "
              f"{'(bit-identical)' if identical else '(DIFFERENT)'}")
        if not identical:
            raise SystemExit("stripped model does not match the original -- do not publish it")

    print("\nStripped model is numerically identical to the original.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
