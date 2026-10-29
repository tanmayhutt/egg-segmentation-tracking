"""Task 2b: segment eggs with hand-written rules, no learning.

Each version is a function image_bgr -> bool mask. Versions are kept so every row of
docs/experiment_log.md can be re-run. Rules were tuned on the dev photos only (config/split.json).

Usage: python scripts/classical.py v1 [dev|eval]
    prints mean IoU against SAM's masks and writes outputs/classical/<version>_<side>.jpg,
    a sheet of the worst photos (our mask in red, SAM's in green, overlap in yellow).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from common import dataset_images, image_number, mask_features
from metrics import evaluate, sam_mask

OUT = Path("outputs/classical")


def largest_blob(binary):
    """Keep only the biggest connected white region."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
    if count <= 1:
        return np.zeros_like(binary, bool)
    biggest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    return labels == biggest


def v1(image):
    """Baseline: grayscale, Otsu's automatic threshold, keep the largest bright blob.

    Otsu picks the threshold that best splits the photo's brightness histogram into two groups.
    Assumes the egg is the brightest large thing in the photo.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return largest_blob(binary > 0)


def v2(image, min_value=0, max_saturation=60, kernel=7):
    """Bright AND colourless, then morphology, then the largest blob.

    Measured on dev: egg saturation sits at 4 to 24, skin and coloured plastic much higher.
    Brightness threshold still from Otsu (per photo). Opening removes specks, closing fills
    holes such as glare spots.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, bright = cv2.threshold(hsv[..., 2], min_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    colourless = hsv[..., 1] <= max_saturation
    binary = ((bright > 0) & colourless).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    return largest_blob(binary > 0)


def candidate_blobs(image, max_saturation=60, kernel=7):
    """v2's bright-and-colourless cleanup, but returning every blob instead of the largest."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, bright = cv2.threshold(hsv[..., 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = ((bright > 0) & (hsv[..., 1] <= max_saturation)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    count, labels = cv2.connectedComponents(binary)
    return [labels == i for i in range(1, count)]


def v3(image, max_saturation=60, kernel=7, min_area=0.005, max_area=0.6,
       min_solidity=0.9, min_ellipse_fill=0.85):
    """Choose blobs by shape, not size. Keeps every egg-shaped blob, so multi-egg photos work.

    solidity      blob area / convex hull area. An egg has no dents; a tray edge or a bag does.
    ellipse_fill  blob area / area of the ellipse fitted to its outline. Eggs are near 1.
    Thresholds start looser than Task 1's SAM filter because thresholded blobs have rougher edges.
    """
    mask = np.zeros(image.shape[:2], bool)
    for blob in candidate_blobs(image, max_saturation, kernel):
        f = mask_features(blob, image)
        if (min_area <= f["area_frac"] <= max_area and f["solidity"] >= min_solidity
                and f["ellipse_fill"] >= min_ellipse_fill):
            mask |= blob
    return mask


def watershed_regions(image, bright_pct=80, max_saturation=60, canny=(30, 100), edge_thick=5,
                      seed_erode=11, background_pct=30, seed_mode="erode", distance_frac=0.7,
                      background_saturation=None):
    """Candidate regions from marker-based watershed. Found on dev to beat plain thresholding.

    1. Seeds: the brightest `bright_pct`-percent pixels that are also colourless, with Canny edge
       pixels removed so an egg and the surface it sits on start as separate seeds. Eroded so
       each seed is safely inside its object.
    2. Background seed: the darkest `background_pct` percent of pixels.
    3. Watershed grows every seed over the unknown pixels until the regions meet at edges.
    Returns one bool mask per seed region.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, *canny),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_thick, edge_thick)))
    seeds = ((v >= np.percentile(v, bright_pct)) & (hsv[..., 1] <= max_saturation) & (edges == 0))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    seeds = cv2.morphologyEx(cv2.morphologyEx(seeds.astype(np.uint8), cv2.MORPH_OPEN, k7), cv2.MORPH_CLOSE, k7)
    if seed_mode == "erode":
        seeds = cv2.erode(seeds, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_erode, seed_erode)))
    else:
        # Distance-transform seeds: keep only the pixels deep inside each blob (at least
        # distance_frac of that blob's deepest point). A fat egg keeps a solid core; a thin bridge
        # leaking into the surface is never deep, so it is cut. Found on dev to recover 86/109
        # eggs against 76 for plain erosion.
        dist = cv2.distanceTransform(seeds, cv2.DIST_L2, 5)
        count, labels = cv2.connectedComponents(seeds)
        core = np.zeros_like(seeds)
        for i in range(1, count):
            blob = labels == i
            core[blob & (dist >= distance_frac * dist[blob].max())] = 1
        seeds = core
    not_egg = v < np.percentile(v, background_pct)
    if background_saturation is not None:
        # Eggs are colourless, so strongly coloured pixels (pink bowl, skin, patterned cloth)
        # are certainly background. Marking them stops egg regions growing into them.
        not_egg |= hsv[..., 1] > background_saturation
    background = cv2.erode(not_egg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    count, markers = cv2.connectedComponents(seeds)
    markers = markers + 1                      # object seeds become 2..count, background 1
    markers[background > 0] = 1
    markers[(seeds == 0) & (background == 0)] = 0   # 0 = unknown, to be flooded
    labels = cv2.watershed(image, markers.astype(np.int32))
    return [labels == i for i in range(2, count + 1)]


def v4(image, min_area=0.005, max_area=0.6, min_solidity=0.9, min_ellipse_fill=0.85, **ws):
    """Watershed regions, then keep the egg-shaped ones."""
    mask = np.zeros(image.shape[:2], bool)
    for region in watershed_regions(image, **ws):
        if region.sum() < 50:
            continue
        f = mask_features(region, image)
        if (min_area <= f["area_frac"] <= max_area and f["solidity"] >= min_solidity
                and f["ellipse_fill"] >= min_ellipse_fill):
            mask |= region
    return mask


def smooth_region(region, kernel=15):
    """Close small gaps and keep the outer outline only, so ragged watershed edges do not spoil
    the shape measurements."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    m = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return out > 0


def v5(image, min_area=0.02, max_area=0.6, min_solidity=0.85, min_ellipse_fill=0.9,
       min_relative_size=0.25, max_overlap=0.5, bright_pct=85, max_saturation=80, **ws):
    """v4 regions, smoothed, with size rules measured on dev.

    min_area 2%: bad regions had median area 1%, good egg regions 9%.
    min_relative_size: a kept region under a quarter of the biggest kept one is a scrap, not an egg.
    max_overlap: regions mostly inside an already-kept egg are duplicates.
    """
    kept = []
    for region in watershed_regions(image, bright_pct=bright_pct, max_saturation=max_saturation, **ws):
        if region.sum() < 50:
            continue
        region = smooth_region(region)
        f = mask_features(region, image)
        if (min_area <= f["area_frac"] <= max_area and f["solidity"] >= min_solidity
                and f["ellipse_fill"] >= min_ellipse_fill):
            kept.append(region)
    mask = np.zeros(image.shape[:2], bool)
    if not kept:
        return mask
    biggest = max(r.sum() for r in kept)
    final = []
    for r in sorted(kept, key=lambda r: -r.sum()):
        if r.sum() >= min_relative_size * biggest and all((r & k).sum() / r.sum() <= max_overlap for k in final):
            final.append(r)
            mask |= r
    return mask


def v6(image, **kw):
    """v5 with distance-transform seeds instead of eroded seeds."""
    return v5(image, seed_mode="distance", **kw)


def v7(image, background_saturation=80, **kw):
    """v6 plus: strongly coloured pixels count as background seeds."""
    return v5(image, seed_mode="distance", background_saturation=background_saturation, **kw)


def trim_leak(region, kernel=31):
    """A region that holds the egg plus a chunk of surface joined by a neck: a large opening cuts
    the neck and removes thin chunks but leaves a fat egg intact. Keep the biggest piece."""
    m = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if count <= 1:
        return None
    return labels == (1 + stats[1:, cv2.CC_STAT_AREA].argmax())


def v8(image, min_area=0.02, max_area=0.6, min_solidity=0.85, min_ellipse_fill=0.9,
       max_ellipse_fill=None, min_relative_size=0.25, max_overlap=0.5, trim_kernel=31, fallback=True,
       min_relative_brightness=None, bright_pct=85, max_saturation=80, background_saturation=60,
       return_regions=False, **ws):
    """Final classical pipeline.

    1. Watershed regions: bright, colourless, edge-cut seeds thinned by the distance transform;
       background seeds from dark or strongly coloured pixels.
    2. Each region is opened with a 31 px ellipse to cut leaks into the surface, then smoothed.
    3. Keep regions that are egg-sized and egg-shaped. Drop scraps and duplicates.
    4. Fallback: every photo in this dataset holds at least one egg, so if nothing passes, keep
       the single most egg-shaped region rather than return an empty mask.
    """
    candidates = []
    for region in watershed_regions(image, bright_pct=bright_pct, max_saturation=max_saturation,
                                    seed_mode="distance", background_saturation=background_saturation, **ws):
        if region.sum() < 50:
            continue
        region = trim_leak(region, trim_kernel) if trim_kernel else region
        if region is None or region.sum() < 50:
            continue
        region = smooth_region(region)
        f = mask_features(region, image)
        if min_area <= f["area_frac"] <= max_area:
            candidates.append((region, f))
    kept = [(r, f) for r, f in candidates if f["solidity"] >= min_solidity and f["ellipse_fill"] >= min_ellipse_fill
            and (max_ellipse_fill is None or f["ellipse_fill"] <= max_ellipse_fill)]
    if not kept and fallback and candidates:
        kept = [max(candidates, key=lambda c: c[1]["solidity"] * c[1]["ellipse_fill"])]
    if kept and min_relative_brightness is not None:
        # Video rule: eggs are the brightest objects in the scene. Walls and reflections that pass
        # the shape rules are much darker than the eggs (measured 140 to 150 against 235).
        top = max(f["brightness"] for _, f in kept)
        kept = [(r, f) for r, f in kept if f["brightness"] >= min_relative_brightness * top]
    mask = np.zeros(image.shape[:2], bool)
    final = []
    if kept:
        biggest = max(r.sum() for r, _ in kept)
        for r, _ in sorted(kept, key=lambda c: -c[0].sum()):
            if r.sum() >= min_relative_size * biggest and all((r & k).sum() / r.sum() <= max_overlap for k in final):
                final.append(r)
                mask |= r
    return final if return_regions else mask


# Settings for the tracking video (Task 4), measured on sample frames. Eggs there cover about 1% of a
# 640x360 frame (photos: 4% to 20%), so the minimum area drops. A white wall corner passed the
# shape rules with ellipse fill 0.96 but brightness 150 against 235 for the eggs, hence the
# relative-brightness rule and the upper fill limit. The fallback is off because title cards hold
# no egg and must return nothing.
VIDEO_SETTINGS = dict(min_area=0.004, max_ellipse_fill=1.1, min_relative_brightness=0.85, fallback=False)


VERSIONS = {"v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5, "v6": v6, "v7": v7, "v8": v8}


def draw(image, pred, ref):
    """Red = ours only, green = SAM only, yellow = both agree."""
    view = image.copy()
    both, ours, sam = pred & ref, pred & ~ref, ref & ~pred
    view[both] = (0.4 * view[both] + 0.6 * np.array([0, 220, 220])).astype(np.uint8)
    view[ours] = (0.4 * view[ours] + 0.6 * np.array([0, 0, 230])).astype(np.uint8)
    view[sam] = (0.4 * view[sam] + 0.6 * np.array([0, 200, 0])).astype(np.uint8)
    return view


def main(version, side):
    segment = VERSIONS[version]
    rows = evaluate(segment, side)
    paths = {image_number(p): p for p in dataset_images()}
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for n in sorted(rows, key=lambda n: rows[n]["iou"])[:16]:
        image = cv2.imread(str(paths[n]))
        tile = cv2.resize(draw(image, segment(image), sam_mask(n)), (240, 240), interpolation=cv2.INTER_AREA)
        label = f"#{n} IoU {rows[n]['iou']:.2f}"
        cv2.rectangle(tile, (0, 0), (10 + 10 * len(label), 22), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    cols = 8
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(OUT / f"{version}_{side}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dev")
