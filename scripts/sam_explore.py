"""Run SAM's automatic mode on a few images and draw every mask it finds, with its measurements.

This is for looking, not for the final result. It shows what SAM proposes before any egg
filter exists, so the filter can be designed from real numbers.

Usage: python scripts/sam_explore.py 1 17 42 ...
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import dataset_images, image_number, mask_features
from sam_model import automatic_generator, load_sam

OUT_DIR = Path("outputs/sam_explore")


def draw_masks(image, masks):
    """Original on the left, every mask in its own colour with its index on the right."""
    rng = np.random.default_rng(0)
    overlay = image.copy()
    for m in masks:
        colour = rng.integers(60, 255, 3)
        overlay[m["segmentation"]] = (0.45 * overlay[m["segmentation"]] + 0.55 * colour).astype(np.uint8)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m["segmentation"])
        x, y = int(xs.mean()), int(ys.mean())
        cv2.putText(overlay, str(i), (x - 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(overlay, str(i), (x - 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.hstack([image, overlay])


def main(numbers):
    sam = load_sam()
    print("device:", sam.device)
    generator = automatic_generator(sam)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(numbers)
    report = {}
    for path in dataset_images():
        n = image_number(path)
        if n not in wanted:
            continue
        image = cv2.imread(str(path))
        masks = generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks.sort(key=lambda m: m["area"], reverse=True)
        rows = []
        for i, m in enumerate(masks):
            f = mask_features(m["segmentation"], image)
            f.update(idx=i, pred_iou=m["predicted_iou"], stability=m["stability_score"])
            rows.append({k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()})
        report[n] = rows
        cv2.imwrite(str(OUT_DIR / f"{n}.jpg"), draw_masks(image, masks))
        print(f"#{n}: {len(masks)} masks")
    (OUT_DIR / "features.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
