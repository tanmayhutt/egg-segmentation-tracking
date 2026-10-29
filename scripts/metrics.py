"""How well does one mask agree with another? Used in Task 2 (tuning on dev) and Task 3 (eval).

All three numbers compare a predicted mask with a reference mask (SAM's), pixel by pixel.

    IoU        overlap / union. 1.0 is a perfect match. The main score.
    precision  of the pixels we called egg, what share really were egg. Low = we grabbed background.
    recall     of the real egg pixels, what share we found. Low = we missed part of the egg.

Pixel accuracy is not used: an egg covers about 5% of a photo, so a mask that says "no egg
anywhere" would score 95% accuracy and be useless.
"""
import json
from pathlib import Path

import cv2
import numpy as np

SPLIT = Path("config/split.json")
SAM_MASKS = Path("outputs/sam_masks")


def scores(pred, ref):
    pred, ref = pred.astype(bool), ref.astype(bool)
    overlap = (pred & ref).sum()
    union = (pred | ref).sum()
    return {
        "iou": overlap / union if union else 1.0,
        "precision": overlap / pred.sum() if pred.sum() else 0.0,
        "recall": overlap / ref.sum() if ref.sum() else 0.0,
    }


def split(side):
    """Photo numbers for 'dev' or 'eval'."""
    return json.loads(SPLIT.read_text())[side]


def sam_mask(n):
    return cv2.imread(str(SAM_MASKS / f"{n}.png"), cv2.IMREAD_GRAYSCALE) > 127


def evaluate(segment, side, verbose=True):
    """Run segment(image_bgr) -> bool mask on every photo of one side; return per-photo scores."""
    from common import dataset_images, image_number
    paths = {image_number(p): p for p in dataset_images()}
    rows = {}
    for n in split(side):
        image = cv2.imread(str(paths[n]))
        rows[n] = scores(segment(image), sam_mask(n))
    if verbose:
        ious = np.array([r["iou"] for r in rows.values()])
        worst = sorted(rows, key=lambda n: rows[n]["iou"])[:8]
        print(f"{side}: mean IoU {ious.mean():.3f}, median {np.median(ious):.3f}, "
              f"photos under 0.5: {(ious < 0.5).sum()}/{len(ious)}")
        print("worst:", ", ".join(f"#{n} {rows[n]['iou']:.2f}" for n in worst))
    return rows
