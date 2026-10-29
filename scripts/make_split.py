"""Task 2a: split the 110 photos into dev (tune the classical rules) and eval (score once) at 4:1,
without leaking a scene across the two sides.

Photos of the same scene are listed by hand in config/scene_groups.json. Automatic grouping
(perceptual hash, colour histogram, SAM embedding) was tried first and failed; see
docs/experiment_log.md. Groups are shuffled with a fixed seed and dealt to eval until eval holds
about one fifth of the photos. The seed is chosen so eval looks like the whole set: a similar share
of multi-egg photos and of photos that needed clicks in Task 1 (the hard ones).

Output:
    config/split.json          dev and eval photo numbers, plus how they were made
    outputs/split_review.jpg   the eval photos, for checking they cover hard cases

Usage: python scripts/make_split.py
"""
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

from common import dataset_images, image_number

GROUPS = Path("config/scene_groups.json")
SUMMARY = Path("outputs/sam_masks/summary.csv")
OUT = Path("config/split.json")
REVIEW = Path("outputs/split_review.jpg")
EVAL_SHARE = 0.2   # 4:1 means one photo in five goes to eval
SEEDS = range(200)


def deal(groups, seed, target):
    """Shuffle the groups and move whole groups to eval until eval is at least `target` photos."""
    order = list(groups)
    random.Random(seed).shuffle(order)
    eval_set, count = [], 0
    for name in order:
        if count >= target:
            break
        eval_set.append(name)
        count += len(groups[name])
    return eval_set


def share(numbers, flagged):
    return sum(n in flagged for n in numbers) / len(numbers)


def main():
    groups = {k: v for k, v in json.loads(GROUPS.read_text()).items() if not k.startswith("_")}
    all_numbers = sorted(n for v in groups.values() for n in v)
    expected = [image_number(p) for p in dataset_images()]
    assert all_numbers == expected, "every photo must be in exactly one group"

    rows = list(csv.DictReader(SUMMARY.open()))
    multi = {int(r["image"]) for r in rows if int(r["eggs"]) > 1}
    hard = {int(r["image"]) for r in rows if r["method"] == "clicks"}
    target = round(EVAL_SHARE * len(all_numbers))

    # Score every seed. Lower is better: eval size near target, and its mix of multi-egg and
    # hard photos near the mix of the whole set.
    best = None
    for seed in SEEDS:
        eval_groups = deal(groups, seed, target)
        ev = sorted(n for g in eval_groups for n in groups[g])
        score = (abs(len(ev) - target) / target
                 + abs(share(ev, multi) - share(all_numbers, multi))
                 + abs(share(ev, hard) - share(all_numbers, hard)))
        if best is None or score < best[0]:
            best = (score, seed, eval_groups, ev)
    score, seed, eval_groups, ev = best
    dev = sorted(set(all_numbers) - set(ev))

    OUT.write_text(json.dumps({
        "method": "scene groups from config/scene_groups.json, whole groups dealt to eval after a "
                  "seeded shuffle; seed chosen from 0..199 to keep eval's mix of multi-egg and "
                  "click-fixed photos close to the whole set",
        "seed": seed,
        "eval_groups": eval_groups,
        "dev": dev,
        "eval": ev,
    }, indent=1))

    print(f"seed {seed}: dev {len(dev)}, eval {len(ev)} (target {target}), ratio {len(dev)/len(ev):.2f}:1")
    print(f"multi-egg share: all {share(all_numbers, multi):.2f}, eval {share(ev, multi):.2f}")
    print(f"click-fixed share: all {share(all_numbers, hard):.2f}, eval {share(ev, hard):.2f}")
    print("eval groups:")
    for g in eval_groups:
        print(f"  {g}: {groups[g]}")

    paths = {image_number(p): p for p in dataset_images()}
    tiles = []
    for n in ev:
        im = cv2.resize(cv2.imread(str(paths[n])), (200, 200), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (60, 22), (0, 0, 0), -1)
        cv2.putText(im, f"#{n}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(im)
    cols = 8
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(REVIEW), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
