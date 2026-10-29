"""Task 3: score the classical segmenter against SAM's masks on the eval photos.

Eval photos were never used while tuning (config/split.json). This script is meant to run once
on eval at the end; it also accepts "dev" for reference.

Output:
    outputs/eval/<version>_<side>.csv      per-photo IoU, precision, recall, scene
    outputs/eval/<version>_<side>.jpg      every photo: ours in red, SAM in green, overlap yellow
    printed summary: mean/median, per-scene table

Usage: python scripts/evaluate.py v8 eval
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from classical import VERSIONS, draw
from common import dataset_images, image_number
from metrics import sam_mask, scores, split

OUT = Path("outputs/eval")
GROUPS = Path("config/scene_groups.json")


def main(version, side):
    segment = VERSIONS[version]
    groups = {k: v for k, v in json.loads(GROUPS.read_text()).items() if not k.startswith("_")}
    scene = {n: g for g, ns in groups.items() for n in ns}
    paths = {image_number(p): p for p in dataset_images()}
    OUT.mkdir(parents=True, exist_ok=True)

    rows, tiles = [], []
    for n in split(side):
        image = cv2.imread(str(paths[n]))
        pred, ref = segment(image), sam_mask(n)
        s = scores(pred, ref)
        rows.append({"image": n, "scene": scene[n], **{k: round(v, 4) for k, v in s.items()}})
        tile = cv2.resize(draw(image, pred, ref), (240, 240), interpolation=cv2.INTER_AREA)
        label = f"#{n} IoU {s['iou']:.2f}"
        cv2.rectangle(tile, (0, 0), (10 + 10 * len(label), 22), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)

    with open(OUT / f"{version}_{side}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    cols = 8
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(OUT / f"{version}_{side}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    ious = np.array([r["iou"] for r in rows])
    print(f"{version} on {side}: {len(rows)} photos")
    print(f"  mean IoU {ious.mean():.3f}, median {np.median(ious):.3f}, "
          f"precision {np.mean([r['precision'] for r in rows]):.3f}, recall {np.mean([r['recall'] for r in rows]):.3f}, "
          f"photos under 0.5: {(ious < 0.5).sum()}, at 0: {(ious == 0).sum()}")
    per = defaultdict(list)
    for r in rows:
        per[r["scene"]].append(r)
    print(f"  {'scene':46} {'n':>2} {'IoU':>5} {'prec':>5} {'rec':>5}")
    for g, rs in sorted(per.items(), key=lambda kv: np.mean([r['iou'] for r in kv[1]])):
        print(f"  {g:46} {len(rs):>2} {np.mean([r['iou'] for r in rs]):.2f}  {np.mean([r['precision'] for r in rs]):.2f}  {np.mean([r['recall'] for r in rs]):.2f}")
    print("  per photo:", ", ".join(f"#{r['image']} {r['iou']:.2f}" for r in rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "eval")
