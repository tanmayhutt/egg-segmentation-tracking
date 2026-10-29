"""Task 1, step A (slow): run SAM's automatic mode on every image and save every mask it finds.

No decision about what is an egg happens here. Saving all candidates means the egg filter in
sam_select.py can be changed and re-run in seconds, without re-running SAM (about 18 s per image).

Output: outputs/sam_candidates/<n>.npz with
    masks      (k, H, W) booleans, packed to bits to save space
    pred_iou   (k,) SAM's own estimate of mask quality
    stability  (k,) how much the mask changes when its cut-off is nudged

Usage: python scripts/sam_candidates.py            (skips images already done)
"""
from pathlib import Path

import cv2
import numpy as np

from common import dataset_images, image_number
from sam_model import automatic_generator, load_sam

OUT_DIR = Path("outputs/sam_candidates")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generator = automatic_generator(load_sam())
    paths = dataset_images()
    for i, path in enumerate(paths, 1):
        out = OUT_DIR / f"{image_number(path)}.npz"
        if out.exists():
            continue
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        masks = generator.generate(image)
        np.savez_compressed(
            out,
            masks=np.packbits(np.stack([m["segmentation"] for m in masks]), axis=-1),
            shape=np.array(image.shape[:2]),
            pred_iou=np.array([m["predicted_iou"] for m in masks]),
            stability=np.array([m["stability_score"] for m in masks]),
        )
        print(f"[{i}/{len(paths)}] #{image_number(path)}: {len(masks)} masks", flush=True)


if __name__ == "__main__":
    main()
