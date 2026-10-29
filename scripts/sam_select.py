"""Task 1, step B (fast): pick which SAM candidate masks are eggs and save one egg mask per image.

SAM does not know what an egg is. It returns masks for everything (eggs, hands, keys, shadows).
This script keeps the masks that look like an egg, using rules measured on sample images
(see scripts/sam_explore.py). Images where the rules fail are fixed with hand-placed clicks
in config/sam_prompts.json, which SAM turns into masks. Those clicks are the only manual input.

Output:
    outputs/sam_masks/<n>.png    white = egg, black = not egg (the Task 1 result)
    outputs/sam_masks/summary.csv  egg count and method (auto / clicks / empty) per image
    outputs/sam_review.jpg       every image with its egg mask drawn, for checking by eye

Usage: python scripts/sam_select.py
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from common import dataset_images, image_number, mask_features

CANDIDATES = Path("outputs/sam_candidates")
PROMPTS = Path("config/sam_prompts.json")
OUT_DIR = Path("outputs/sam_masks")
REVIEW = Path("outputs/sam_review.jpg")

# Rules for "this mask is an egg". Each was checked against real numbers from sample images.
MIN_AREA, MAX_AREA = 0.008, 0.6  # drop specks and whole-background masks
MIN_BRIGHTNESS = 120             # eggs measured 137+; shadows, keys, ash measured under 120
MAX_SATURATION = 45              # eggs are nearly colourless (measured 4 to 33)
MIN_SOLIDITY = 0.95              # an egg outline has no dents (hands and fingers do)
MIN_ELLIPSE_FILL = 0.96          # eggs measured 0.97+; empty egg-tray cups 0.90 to 0.95
MIN_ONE_PIECE = 0.95             # the largest connected piece must hold almost all the mask
MIN_RELATIVE_SIZE = 0.25         # drop egg-like blobs far smaller than the biggest egg (earbuds)
MAX_OVERLAP = 0.5                # a mask mostly inside an already-kept egg is a duplicate or part


def load_candidates(n):
    data = np.load(CANDIDATES / f"{n}.npz")
    h, w = data["shape"]
    masks = np.unpackbits(data["masks"], axis=-1, count=w).astype(bool)
    return masks, data["pred_iou"]


def largest_piece_share(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        return 0.0
    return stats[1:, cv2.CC_STAT_AREA].max() / mask.sum()


def looks_like_egg(f, mask):
    return (MIN_AREA <= f["area_frac"] <= MAX_AREA
            and f["brightness"] >= MIN_BRIGHTNESS
            and f["saturation"] <= MAX_SATURATION
            and f["solidity"] >= MIN_SOLIDITY
            and f["ellipse_fill"] >= MIN_ELLIPSE_FILL
            and largest_piece_share(mask) >= MIN_ONE_PIECE)


def select_eggs(image, masks, pred_iou):
    """Return a list of egg masks chosen from SAM's candidates."""
    eggs = [(m, s) for m, s in zip(masks, pred_iou) if looks_like_egg(mask_features(m, image), m)]
    if not eggs:
        return []
    biggest = max(m.sum() for m, _ in eggs)
    eggs = [(m, s) for m, s in eggs if m.sum() >= MIN_RELATIVE_SIZE * biggest]

    # Most confident first; skip any mask that mostly overlaps an egg already kept.
    kept = []
    for m, _ in sorted(eggs, key=lambda e: -e[1]):
        if all((m & k).sum() / m.sum() <= MAX_OVERLAP for k in kept):
            kept.append(m)
    return kept


def eggs_from_clicks(predictor, image, eggs_clicks, pick="best"):
    """One SAM mask per egg. Each egg is a list of [x, y] clicks placed on it.

    One click usually works. An egg split by something in front of it (a rod) gets a click
    on each side, so SAM treats both halves as one object.
    pick="smallest" keeps SAM's smallest guess instead of its most confident one. Used where
    the egg is blurred (boiling water) and the most confident guess is the whole bowl.
    """
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    eggs = []
    for clicks in eggs_clicks:
        points = np.array(clicks)
        # With one click SAM is unsure how big the object is, so ask for 3 guesses and keep the
        # most confident. With several clicks the object is clear, so one guess is enough.
        single = len(points) == 1
        masks, scores, _ = predictor.predict(point_coords=points, point_labels=np.ones(len(points)),
                                             multimask_output=single)
        if pick == "smallest":
            eggs.append(min(masks, key=lambda m: m.sum()))
        else:
            eggs.append(masks[scores.argmax()])
    return eggs


def draw_review(image, mask, n, count, method):
    view = image.copy()
    view[mask] = (0.5 * view[mask] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (0, 255, 0), 3)
    view = cv2.resize(view, (200, 200), interpolation=cv2.INTER_AREA)
    tag = {"auto": "", "clicks": " C", "empty": " E"}[method]
    label = f"{n}: {count}{tag}"
    cv2.rectangle(view, (0, 0), (8 + 11 * len(label), 22), (0, 0, 0), -1)
    cv2.putText(view, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return view


def main():
    prompts = json.loads(PROMPTS.read_text()) if PROMPTS.exists() else {}
    predictor = None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []

    for path in dataset_images():
        n = image_number(path)
        image = cv2.imread(str(path))
        fix = prompts.get(str(n))
        if fix is None:
            eggs, method = select_eggs(image, *load_candidates(n)), "auto"
        elif fix.get("empty"):
            eggs, method = [], "empty"
        else:
            if predictor is None:
                from sam_model import load_sam, point_predictor
                predictor = point_predictor(load_sam())
            eggs = eggs_from_clicks(predictor, image, fix["eggs"], fix.get("pick", "best"))
            method = "clicks"

        mask = np.zeros(image.shape[:2], bool)
        for egg in eggs:
            mask |= egg
        cv2.imwrite(str(OUT_DIR / f"{n}.png"), mask.astype(np.uint8) * 255)
        rows.append({"image": n, "eggs": len(eggs), "method": method,
                     "note": fix.get("note", "") if fix else ""})
        tiles.append(draw_review(image, mask, n, len(eggs), method))

    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    cols = 10
    blank = np.full_like(tiles[0], 255)
    tiles += [blank] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(REVIEW), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    methods = [r["method"] for r in rows]
    print(f"{len(rows)} images: {methods.count('auto')} auto, {methods.count('clicks')} clicks, "
          f"{methods.count('empty')} empty; images with 0 eggs: "
          f"{[r['image'] for r in rows if r['eggs'] == 0]}")


if __name__ == "__main__":
    main()
