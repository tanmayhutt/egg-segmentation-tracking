"""Tile every dataset image into one picture so the whole dataset can be seen at once.

Images are sorted by the number in their original filename, so photos taken one after
another sit next to each other. That makes near-duplicates (a leakage risk) easy to spot.

Usage: python scripts/contact_sheet.py data/egg outputs/contact_sheet.jpg
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

TILE = 150   # size of each thumbnail in pixels
COLS = 11    # thumbnails per row
SPLIT_TAG = {"train": "tr", "valid": "va", "test": "te"}


def image_number(path):
    """'not_damaged_101_jpg.rf.<hash>.jpg' -> 101"""
    match = re.search(r"_(\d+)_jpg", path.name)
    return int(match.group(1)) if match else -1


def main(data_dir, out_path):
    paths = sorted(Path(data_dir).glob("*/*/*.jpg"), key=image_number)
    rows = (len(paths) + COLS - 1) // COLS
    sheet = np.full((rows * TILE, COLS * TILE, 3), 255, dtype=np.uint8)

    for i, path in enumerate(paths):
        thumb = cv2.resize(cv2.imread(str(path)), (TILE, TILE), interpolation=cv2.INTER_AREA)
        split = SPLIT_TAG.get(path.parts[-3], "?")
        if path.parent.name == "Unlabeled":
            split = "un"
        label = f"{image_number(path)} {split}"
        # Black box behind the label so it is readable on any photo.
        cv2.rectangle(thumb, (0, 0), (70, 20), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        r, c = divmod(i, COLS)
        sheet[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = thumb

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(paths)} images -> {out_path} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
