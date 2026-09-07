"""One-time extraction of real BTSbot cutout images for the UI's Object
Detail tab (CLAUDE.md's own Day-10 note: "images_v10.zip cutout archive is
out of scope this session" -- that decision was about the IMAGE ENCODER
[training a model on pixels, §10's "cut, not deferred"]. Merely DISPLAYING
a real object's own cutout in the UI is presentation only, a completely
different scope, and was never decided against.

`data/raw/btsbot/images_v10.zip` holds one `.npy` per split.
`test_triplets_v10_N100_programid1.npy` is `(49732, 63, 63, 3)` float64 --
49,732 rows, which matches `test_cand_v10_N100_programid1.csv`'s row count
(49733 lines - 1 header = 49732) exactly. Row `i` of the array is therefore
assumed to correspond to row `i` of that split's metadata CSV, in file
order -- verified only by this row-count match, not by any documented
id column in the archive itself (there is none). Channel order `[sci, ref,
diff]` is the standard ZTF alert-cutout convention (matches
`src.serve.cutouts.PLANE_NAMES`) but is likewise an assumption, not
independently confirmed against BTSbot's own docs.

Multiple CSV rows can share one `objectId` (ZTF re-alerts an object on
every new detection); this script keeps only the LATEST (max `jd`) row per
object, matching `src.serve.cutouts`' own "latest triplet only" convention.

The zip's `test` entry alone is ~4.7GB uncompressed -- too large to decode
per-request in a live server, and this machine has only ~16GB free disk
(§5.1-style constraint, not a hard rule anywhere else). This script
extracts the compressed entry to a temporary local `.npy` (memory-mapped,
so the ~1994 needed rows are paged in without loading the full array into
RAM), renders each kept row's 3 planes through the same ZScale helper
`src.serve.cutouts` already uses for live Fink cutouts, and deletes the
temporary file when done -- output is one small JSON triplet per object
under `data/interim/btsbot_cutouts/{split}/{object_id}.json`, matching
`cutouts_to_png_triplet`'s own `{"science": ..., "template": ...,
"difference": ...}` shape exactly so `src.serve.api` can serve it through
the same response shape as a live-Fink cutout.

Entry point: `python -m src.serve.extract_btsbot_cutouts [--split test]`
"""

import argparse
import csv
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from src.serve.cutouts import cutouts_to_png_triplet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IMAGES_ZIP = Path("data/raw/btsbot/images_v10.zip")
METADATA_DIR = Path("data/raw/btsbot/metadata_v10")
OUT_DIR = Path("data/interim/btsbot_cutouts")

_ARRAY_NAME = {
    "train": "train_triplets_v10_N100_programid1.npy",
    "val": "val_triplets_v10_N100_programid1.npy",
    "test": "test_triplets_v10_N100_programid1.npy",
}
_CSV_NAME = {
    "train": "train_cand_v10_N100_programid1.csv",
    "val": "val_cand_v10_N100_programid1.csv",
    "test": "test_cand_v10_N100_programid1.csv",
}


def _latest_row_per_object(csv_path: Path) -> dict[str, int]:
    """objectId -> row index (0-based, matching array row order) of that
    object's max-`jd` CSV row."""
    best_row: dict[str, int] = {}
    best_jd: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        obj_idx = header.index("objectId")
        jd_idx = header.index("jd")
        for row_idx, row in enumerate(reader):
            object_id = row[obj_idx]
            jd = float(row[jd_idx])
            if object_id not in best_jd or jd > best_jd[object_id]:
                best_jd[object_id] = jd
                best_row[object_id] = row_idx
    return best_row


def extract(split: str = "test") -> int:
    csv_path = METADATA_DIR / _CSV_NAME[split]
    logger.info("Scanning %s for latest row per object_id...", csv_path)
    latest_row = _latest_row_per_object(csv_path)
    logger.info("%d distinct objects in %s split", len(latest_row), split)

    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(IMAGES_ZIP) as z:
        member = _ARRAY_NAME[split]
        logger.info("Extracting %s to a local temp file (one-time decompression)...", member)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(z.extract(member, path=tmp_dir))
            logger.info("Decompressed to %s (%.1f GB) -- memory-mapping", tmp_path, tmp_path.stat().st_size / 1e9)
            arr = np.load(tmp_path, mmap_mode="r")
            logger.info("Array shape %s dtype %s", arr.shape, arr.dtype)

            n_written = 0
            for object_id, row_idx in latest_row.items():
                out_path = out_dir / f"{object_id}.json"
                if out_path.exists():
                    n_written += 1
                    continue
                triplet_hwc = np.asarray(arr[row_idx])  # (63, 63, 3), copies out of the mmap
                triplet_chw = np.transpose(triplet_hwc, (2, 0, 1))  # -> (3, 63, 63): sci, ref, diff
                images = cutouts_to_png_triplet(triplet_chw)
                with open(out_path, "w") as f:
                    json.dump({"object_id": object_id, "images": images}, f)
                n_written += 1
                if n_written % 200 == 0:
                    logger.info("Rendered %d/%d objects...", n_written, len(latest_row))
            del arr  # release the mmap before the temp dir is cleaned up

    logger.info("Done: %d cutout triplets written to %s", n_written, out_dir)
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    extract(split=args.split)


if __name__ == "__main__":
    main()
