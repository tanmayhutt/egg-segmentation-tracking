"""Task 4: track each egg separately through the video.

Detect: v8 with VIDEO_SETTINGS on every frame (resized to 640 wide, the scale the rules were tuned at).
Track:  match detections to existing tracks by centre distance with the Hungarian algorithm
        (scipy.optimize.linear_sum_assignment), so each egg keeps one ID over time.
        - a track with no match survives MAX_MISSED frames on its last position (hand passing over)
        - a detection that matches no live track is first offered to a hidden track with no distance
          limit: eggs do not appear from nowhere, so a new egg while one is missing is that egg,
          carried by the hand
        - a new ID is only confirmed after MIN_HITS consecutive detections (no IDs for wrists or fades)
        - a scene cut (big change between consecutive frames) drops every track

Output:
    outputs/video/tracked.mp4   each egg outlined in its own colour with ID and trail
    outputs/video/tracks.csv    frame, time_s, id, cx, cy, area_px

Usage: python scripts/track.py [data/video/eggs.mp4]
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from classical import VIDEO_SETTINGS, v8

OUT = Path("outputs/video")
WIDTH = 640          # process at this width; draw at full size
MAX_DISTANCE = 60    # px at 640 wide: a match farther than this is not the same egg
MAX_MISSED = 120     # frames (5 s at 24 fps) a track survives without a detection; the hand hides a spinning egg for up to 3.7 s
MIN_HITS = 12        # consecutive detections (0.5 s) before a track gets an ID
SCENE_CUT = 40       # mean absolute pixel change between frames that counts as a cut
COLOURS = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (200, 0, 255), (0, 0, 255), (255, 255, 0)]


class Track:
    def __init__(self, centre, area, contour):
        self.centre, self.area, self.contour = centre, area, contour
        self.hits, self.missed, self.id = 1, 0, None
        self.trail = [centre]

    def update(self, centre, area, contour):
        self.centre, self.area, self.contour = centre, area, contour
        self.hits += 1
        self.missed = 0
        self.trail.append(centre)
        self.trail = self.trail[-72:]   # keep 3 s of history


def detect(frame_small):
    """One (centre, area, contour) per egg region."""
    out = []
    for region in v8(frame_small, return_regions=True, **VIDEO_SETTINGS):
        ys, xs = np.nonzero(region)
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.append(((xs.mean(), ys.mean()), len(xs), max(contours, key=cv2.contourArea)))
    return out


def assign(track_idx, det_idx, tracks, detections, max_distance):
    """Hungarian assignment on centre distance between the given tracks and detections."""
    if not track_idx or not det_idx:
        return []
    cost = np.array([[np.hypot(tracks[t].centre[0] - detections[d][0][0], tracks[t].centre[1] - detections[d][0][1])
                      for d in det_idx] for t in track_idx])
    rows, cols = linear_sum_assignment(cost)
    return [(track_idx[r], det_idx[c]) for r, c in zip(rows, cols) if cost[r, c] <= max_distance]


def match(tracks, detections):
    """Two rounds: nearby matches first, then hidden tracks may claim far-away leftovers."""
    pairs = assign(list(range(len(tracks))), list(range(len(detections))), tracks, detections, MAX_DISTANCE)
    used_t, used_d = {r for r, _ in pairs}, {c for _, c in pairs}
    hidden = [k for k, t in enumerate(tracks) if k not in used_t and t.missed > 0 and t.id is not None]
    leftover = [c for c in range(len(detections)) if c not in used_d]
    for r, c in assign(hidden, leftover, tracks, detections, np.inf):
        # Same egg, same size: refuse a re-link to something half or double the egg's area
        # (a brightening wall during a fade passed the shape rules once).
        if 0.5 <= detections[c][1] / tracks[r].area <= 2.0:
            pairs.append((r, c))
    return pairs


def main(video):
    cap = cv2.VideoCapture(video)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = w / WIDTH
    OUT.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rows, tracks, next_id, history = [], [], 1, []

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (WIDTH, int(h / scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
        # Scene change: compare with the previous frame (hard cut) and with the frame 0.5 s ago
        # (a fade). Either one large enough means nothing carries over.
        history.append(gray)
        history = history[-13:]
        if len(history) > 1 and max(np.abs(gray - history[-2]).mean(), np.abs(gray - history[0]).mean()) > SCENE_CUT:
            tracks = []
        detections = detect(small)

        pairs = match(tracks, detections)
        matched_t, matched_d = {r for r, _ in pairs}, {c for _, c in pairs}
        for r, c in pairs:
            tracks[r].update(*detections[c])
        for t in (t for k, t in enumerate(tracks) if k not in matched_t):
            t.missed += 1
            t.hits = 0
        for c, d in enumerate(detections):
            if c not in matched_d:
                tracks.append(Track(*d))
        for t in tracks:
            if t.id is None and t.hits >= MIN_HITS:
                t.id, next_id = next_id, next_id + 1
        tracks = [t for t in tracks if t.missed <= MAX_MISSED]

        for t in tracks:
            if t.id is None:
                continue
            colour = COLOURS[(t.id - 1) % len(COLOURS)]
            cx, cy = int(t.centre[0] * scale), int(t.centre[1] * scale)
            if t.missed == 0:
                cv2.drawContours(frame, [(t.contour * scale).astype(np.int32)], -1, colour, 3)
                rows.append({"frame": i, "time_s": round(i / fps, 3), "id": t.id, "cx": cx, "cy": cy,
                             "area_px": int(t.area * scale * scale)})
            else:
                cv2.circle(frame, (cx, cy), 12, colour, 2)   # dashed-ish marker: egg hidden, position held
            pts = (np.array(t.trail) * scale).astype(np.int32)
            cv2.polylines(frame, [pts], False, colour, 2)
            label = f"egg {t.id}" + ("" if t.missed == 0 else " (hidden)")
            cv2.putText(frame, label, (cx - 30, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(frame, label, (cx - 30, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.putText(frame, f"frame {i}  eggs tracked: {sum(t.id is not None for t in tracks)}", (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        if i % 240 == 0:
            print(f"frame {i}/{n}: {len(detections)} detections, ids {[t.id for t in tracks if t.id]}", flush=True)

    writer.release()
    with open(OUT / "tracks.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["frame", "time_s", "id", "cx", "cy", "area_px"])
        wr.writeheader()
        wr.writerows(rows)
    ids = sorted({r["id"] for r in rows})
    print(f"done: {len(rows)} detections, {len(ids)} ids: {ids}")
    for k in ids:
        fr = [r["frame"] for r in rows if r["id"] == k]
        print(f"  egg {k}: frames {fr[0]} to {fr[-1]} ({fr[0]/fps:.1f}s to {fr[-1]/fps:.1f}s), seen in {len(fr)} frames")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/video/eggs.mp4")
