"""Shared helpers: finding dataset images and measuring the shape and colour of a mask."""
import re
from pathlib import Path

import cv2
import numpy as np


def image_number(path):
    """'not_damaged_101_jpg.rf.<hash>.jpg' -> 101"""
    match = re.search(r"_(\d+)_jpg", Path(path).name)
    return int(match.group(1)) if match else -1


def dataset_images(data_dir="data/egg"):
    """All dataset images, sorted by their original number."""
    return sorted(Path(data_dir).glob("*/*/*.jpg"), key=image_number)


def mask_features(mask, image_bgr):
    """Numbers that describe one mask. Used to decide whether a mask is an egg.

    area_frac     share of the image the mask covers (0 to 1)
    brightness    mean HSV value inside the mask (0 to 255). Eggs are bright.
    saturation    mean HSV saturation inside the mask (0 to 255). Eggs are nearly colourless.
    solidity      mask area / convex hull area. 1.0 means no dents, like an egg.
    ellipse_fill  mask area / area of the ellipse fitted to its outline. About 1.0 for an egg.
    """
    mask = mask.astype(np.uint8)
    area = int(mask.sum())
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    inside = mask.astype(bool)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    ellipse_area = 0.0
    if len(largest) >= 5:  # fitEllipse needs at least 5 points
        (_, _), (w, h), _ = cv2.fitEllipse(largest)
        ellipse_area = np.pi * w * h / 4

    return {
        "area_frac": area / mask.size,
        "brightness": float(hsv[..., 2][inside].mean()),
        "saturation": float(hsv[..., 1][inside].mean()),
        "solidity": area / hull_area if hull_area else 0.0,
        "ellipse_fill": area / ellipse_area if ellipse_area else 0.0,
    }
