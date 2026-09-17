#!/usr/bin/env python3
"""Download the published model weights and verify their checksums.

Weights are distributed separately from the code, on the HuggingFace Hub,
because they are ~133 MB each and carry their own licence (CC-BY-4.0).

    python scripts/download_weights.py --dest weights/
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

# --- Published record identifiers ---------------------------------------
HF_REPO = "yansonng/ivcmassist"

MODELS = {
    "mosaic_nerves_v1.keras": {
        "sha256": "f9e6bdca92b877e92f4ed8b609ae0ec36519e04f93744c84d5dabf535e33e197",
        "description": "Nerve segmentation v1 (637 mosaic crops), 2 classes",
    },
    "mosaic_nerves_v2.keras": {
        "sha256": "2f4b2f9ec62fd0f37567505021d87db435af02ae91e59b56285f4e7fb6badde1",
        "description": "Nerve segmentation v2 (16,313 mosaic crops), 2 classes",
    },
    "mosaic_dc.keras": {
        "sha256": "d91a188d3c69094f9686221f52685fc643ac803073480da2d9f5c887fc43b58c",
        "description": "Dendritic cell segmentation, 3 classes (shared by v1 and v2)",
    },
}
# ---------------------------------------------------------------------------


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"  fetching {url}")

    def progress(count, block_size, total):
        if total > 0:
            pct = min(100, 100 * count * block_size / total)
            print(f"\r  {pct:5.1f}%  ({total / 1e6:.0f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, default=Path("weights"))
    p.add_argument("--model", choices=sorted(MODELS), action="append",
                   help="download only this model (repeatable; default: all)")
    p.add_argument("--verify-only", action="store_true",
                   help="checksum files already in --dest without downloading")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    wanted = args.model or sorted(MODELS)
    failures = 0

    for name in wanted:
        meta = MODELS[name]
        target = args.dest / name
        print(f"\n{name} -- {meta['description']}")

        if not target.exists() and not args.verify_only:
            download(f"https://huggingface.co/{HF_REPO}/resolve/main/{name}?download=true", target)

        if not target.exists():
            print("  missing")
            failures += 1
            continue

        actual = sha256sum(target)
        if actual == meta["sha256"]:
            print(f"  sha256 OK  {actual}")
        else:
            print(f"  CHECKSUM MISMATCH\n    expected {meta['sha256']}\n    got      {actual}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
