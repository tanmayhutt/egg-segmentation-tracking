# Experiment log

Every trial, in order, including the ones that failed. For each: what changed, why, what happened,
and what was decided. Numbers are measured, not guessed.

## Split (Task 2a): grouping same-scene photos

Goal: put photos of the same scene on one side of the dev/eval split, so rules tuned on dev are not
tested on a near-copy. Test for any grouping method: the known same-scene sets (keyboard #17/18/19/43,
steel bowl #113/114/128/129) must be closer to each other than unrelated photos are.

| # | Method | Known same-scene pairs | Unrelated nearest neighbours | Result |
|---|---|---|---|---|
| 1 | Perceptual hash (`imagehash.phash`, 64 bits), Hamming distance | 18 to 34 bits apart | 16 to 24 bits apart | Failed. Same-scene photos are shot from different angles, so the light/dark layout differs |
| 2 | HSV colour histogram (8x8x8 bins), cosine distance | 0.09 to 0.34 | 0.0 to 0.3 | Failed. Nearly every photo is a white egg on a grey or white surface, so histograms all look alike |
| 3 | SAM image embedding, averaged to 256 numbers, cosine distance | 0.003 to 0.018 | 0.00 to 0.02 | Failed, same reason as 2. Averaging throws away where things are |

Also checked: no capture metadata survives (all 110 files have zero EXIF bytes; Roboflow's readme confirms it stripped EXIF), and the photo numbers do not follow scenes (20 of the 24 scenes are spread over non-adjacent numbers, e.g. keyboard #17/18/19 and #43). So no existing grouping could be reused.

Decision: group by eye from `outputs/contact_sheet.jpg` into `config/scene_groups.json`, then split
whole groups 4:1 with a fixed random seed (`scripts/make_split.py`). 110 photos is small enough to do
this honestly. A deep scene model (CLIP, DINOv2) would likely work but adds a download and a
dependency for a one-off step, so it was not tried.

Result of the split: seed 21, dev 88, eval 22 (4.00:1). Multi-egg share 0.15 overall vs 0.23 in eval;
click-fixed share 0.13 vs 0.14. Eval scenes: striped towel, ash pile, steel bowl (incl. boiling water),
brown marble, red lid. Eval has no hand photos and no beige-marble counter photos.

## Classical segmenter (Task 2b), tuned on the 88 dev photos only

Score: mean IoU against SAM's masks over dev (`scripts/metrics.py`). Eval is never touched here.

### Measurement before any rule

Per-photo median of egg pixels vs background pixels (using SAM's dev masks), 10th to 90th percentile:

| Channel | Egg | Background | Photos where egg > background |
|---|---|---|---|
| Gray | 168 to 209 | 53 to 173 | 98% |
| HSV V | 172 to 215 | 61 to 186 | 97% |
| Lab L | 175 to 214 | 55 to 181 | 98% |
| HSV S | 4 to 24 | 7 to 123 | 9% (eggs are less saturated) |
| HSV H | 13 to 110 | 16 to 116 | 51% (useless) |

Reading: brightness is the main signal, low saturation the second. Hue carries nothing. The ranges
overlap (brightest backgrounds reach the darkest eggs), so a single fixed threshold cannot work for
every photo; the threshold has to be chosen per photo.

### Trials

| # | Version | What changed and why | Dev mean IoU | Under 0.5 | What happened |
|---|---|---|---|---|---|
| 1 | v1 | Baseline: grayscale, Otsu threshold, keep the largest bright blob | 0.371 | 61/88 | Two failures. Hands (#1, #2, #20, #24): skin passes the threshold and is bigger than the egg, so the largest blob is the hand. Bright surfaces (white bag, tray, paper, pink bowl): egg and surface merge into one blob |
| 2 | v2 | Pixels must be bright (Otsu on HSV V) AND colourless (S <= 60); opening and closing with a 7 px ellipse; largest blob | 0.425 | 55/88 | Better: hands in normal light now rejected (skin is saturated). Still fails where another bright colourless object is bigger than the egg: steel tray (#105), white label (#22), white bag (#125), egg tray (#42, #116). Backlit hands (#20, #24) are hopeless for any brightness rule: the egg (V 50) is darker than the background (V 90) |
| 3 | v3 | Keep every blob by shape (solidity >= 0.9, ellipse fill >= 0.85) instead of the largest | 0.283 | 59/88 | Worse, median 0. Diagnosis on dev: in 51 of 88 photos the egg is already merged into a bigger bright blob after Otsu (white bag, marble, towel), and no shape rule can split a merged blob. 12 more had an egg blob whose rough thresholded edge failed solidity 0.9. The thresholding step, not the filter, is the bottleneck |

### Side experiments on the thresholding step (dev only)

Question: after thresholding and cleanup, does the egg exist as its own blob? Scored first by "best blob
IoU with SAM's mask >= 0.5" per photo. That score was later found to be wrong for multi-egg photos (one
region can never match a three-egg mask), so the last table scores each SAM egg against its best region.

| Binarisation | Photos where the egg survives as its own blob |
|---|---|
| Otsu on V | 37/88 (41 with a 15 px opening) |
| Otsu applied twice (brighter half of the bright class) | 38/88 |
| Brightest 20% of pixels | 45/88 |
| Brightest 10% / 5% | 33 / 7 (the egg's shaded side is lost) |
| Local contrast, V minus 101 px median blur | 0 to 3/88. Window smaller than the egg, so the egg became its own background |
| Local contrast, V minus Gaussian sigma 60 | 42/88 |
| Brightest 20% with Canny edges (30/100, 5 px) removed | 43/88 |
| Watershed: seeds = eroded brightest-20% blobs, background seed = darkest 50% | 51/88 |
| Watershed with edge-cut seeds and background seed = darkest 30% | 55/88 |

Per-egg scoring (each SAM egg vs its best region; photo counts when every egg >= 0.5):

| Binarisation | Photos | Mean per-egg best IoU |
|---|---|---|
| Edge-cut blobs only, holes filled, grown 3 px, brightest 20% | 47/88 | 0.591 |
| Watershed, edge-cut seeds, brightest 20%, bg darkest 30% | **54/88** | **0.669** |
| Same, brightest 40% | 42/88 | 0.558 |
| Same, brightest 60% | 30/88 | 0.451 |

Looser brightness lets the surface into the seeds and merges again. Remaining failures from the debug
sheet (`outputs/classical/ws_debug.jpg`): regions bleed on the egg's shaded side where it meets a
surface of similar tone (#110 wall, #104 bag), the backlit hands (#20, #24, #44) where the egg is
darker than the background, and the egg inside a white sticker on a pink bowl (#34).
| 4 | v4 | Watershed regions (edge-cut seeds, brightest 20%, background darkest 30%), then v3's shape filter | 0.350 | 55/88 | Median still 0. Grid over seed and shape thresholds on dev: best 0.454 at brightest 15%, saturation <= 80, solidity >= 0.85, ellipse fill >= 0.9. Far below the 0.67 per-egg ceiling of the regions themselves, so the selection step is losing good regions and admitting bad ones |

Measurement on dev (v4 regions, brightest 15%, saturation <= 80): 78 regions match an egg (IoU >= 0.5),
732 do not. Shape separates them only partly: 79% of good and 21% of bad pass solidity >= 0.85 and
fill >= 0.9. Size separates them better: bad regions have median area 1% of the photo, good regions
9%, and only 10% of good regions are under 4%. The minimum area of 0.5% was letting hundreds of small
bright scraps through.
| 5 | v5 | v4 regions smoothed (close 15 px, outer outline only); min area 2%; drop regions under a quarter of the biggest kept; drop regions mostly inside a kept one | 0.465 | 46/88 | First version with a non-zero median (0.436). Breakdown of the 46 failures on dev: 26 no region matches the egg, 10 a matching region existed but failed the shape rules, 8 a wrong region was kept, 2 boundary or missing eggs |

Why those fail (measured): in both the "no match" and "shape rules" buckets the best region has recall
near 1.0 and precision 0.3 to 0.6. It holds the whole egg plus a chunk of the surface next to it. The
egg's outline is mostly found by Canny (only 2 to 4% of egg pixels fall on an edge) but the edge has
gaps on the shaded side, and the seed leaks through the gap into the surface. A second candidate
source, shapes enclosed by closed Canny contours, recovered only 6 of 109 dev eggs on its own and 3
more in union with watershed, so it was dropped.
| 6 | v6 | Seeds from the distance transform (pixels at >= 70% of each blob's depth) instead of 11 px erosion. Side test on dev: eggs with a matching region 76 -> 86 of 109, mean best IoU 0.684 -> 0.733. Erosion 21/31 px and a gradient-image watershed did not help | 0.450 | 45/88 | Median up (0.436 -> 0.452) but mean down slightly. Better candidates, so the selection rules are now the bottleneck: they were tuned for v4's regions |

Grid over the selection rules on cached v6 regions (min area 1/2/4%, solidity 0.8 to 0.95, ellipse fill
0.8 to 0.95, relative size 0.25/0.5, 96 combinations): best 0.477 (min area 1%, solidity 0.8, fill 0.9),
current 0.450. Thresholds are not the lever. v6 failure breakdown: 20 generation, 13 selection miss,
11 false positive, 1 partial.
| 7 | v7 | Strongly coloured pixels (S > 60) added to the background seed, since eggs are colourless. Also tried S > 80 and S > 100 | 0.451 | 42/88 | Median 0.452 -> 0.616: half the dev photos now score well. Mean unchanged because 42 photos still score exactly 0, so the score is bimodal. S > 60 kept |
| 8 | v8 | Each region opened with a 31 px ellipse to cut leaks into the surface (21 and 41 also tried); fallback keeps the single most egg-shaped region when nothing passes, since every photo holds an egg | **0.524** | 36/88 | Zero-score photos 37 -> 15. Opening alone 0.510, fallback alone 0.502, both 0.524. Kept as the final classical pipeline |

## Eval (Task 3), run once

`python scripts/evaluate.py v8 eval` and `v1 eval`. 22 photos, never used during tuning.

| Version | Mean IoU | Median | Precision | Recall | Under 0.5 | At 0 |
|---|---|---|---|---|---|---|
| v1 baseline | 0.188 | 0.168 | 0.206 | 0.605 | 20/22 | 7 |
| v8 final | **0.472** | 0.452 | 0.544 | 0.635 | 12/22 | 5 |

Dev was 0.524, eval 0.472: a drop of 0.05, which is the usual sign of mild over-tuning to dev, not a
collapse. The baseline shows the tuning was real on unseen photos (0.19 -> 0.47).

Per scene on eval (v8): steel bowl 0.66 (10 photos, the three boiling-water ones 0.00 / 0.22 / 0.69),
brown marble 0.60, striped towel 0.36, red lid 0.27, ash pile 0.25 (6 photos, three at 0.00).

Failure analysis from `outputs/eval/v8_eval.jpg` (ours red, SAM green, agreement yellow):

- **Ash pile (0.25).** Three of six photos at 0.00 (#27, #50, #78): the egg is dusted with ash, so it
  has the same grey tone as the pile. No brightness contrast, no colour contrast, no edge. The
  fallback then keeps a light ash patch instead. Where the egg is clean (#41, 0.83) it works.
- **Striped towel (0.36).** White cloth is bright and colourless, like the egg. Towel patches pass
  the shape rules and are kept alongside the egg (low precision), and in #4 a large egg filling the
  frame is missed entirely.
- **Steel bowl (0.66).** Clear water: 0.87 to 0.98 (#113, #114, #129, #159). Boiling water: 0.00 and
  0.22 (#135, #124), steam and bubbles blur the outline; SAM's own masks there were the least certain
  in Task 1. Bright steel rims and reflections are kept as extra regions in #128, #147 and #124.
- **Boundary error everywhere.** Our masks under-cover the egg's shaded side (green crescents on #3,
  #6, #36, #129): those pixels fall below the brightness cut, and watershed stops at the shading
  gradient. This is the main reason recall (0.64) is not higher even on found eggs.
- **Red lid (#31, 0.27).** Egg found, but the bright tiled floor behind it joins the region.

Same root cause as on dev: the method has three cues, brightness, colourlessness and edges, and the
failures are exactly the scenes where the surface shares all three with the egg.

## Video (Task 4): adapting v8 to the frames

Checked on 16 sample frames of `data/video/eggs.mp4` at 640x360, with the fallback off (title cards
must return nothing).

| # | Change | Why | Result on the 16 frames |
|---|---|---|---|
| 1 | v8 as is, fallback off | Baseline | Only one of two eggs found in most frames; a white wall corner kept in 3 frames; a sleeve in 1 |
| 2 | Minimum area 2% -> 0.4% | Measured: the two eggs are 0.0202 and 0.0199 of the frame, one each side of the 2% cut | Both eggs found, but 3 to 5 regions per frame: wall, shirt, egg reflections on the glossy table |
| 3 | Ellipse fill capped at 1.1 | A rounded rectangle over-fills its fitted ellipse; eggs sit at 0.96 to 1.03 | Little change alone; the wall corner measured 0.96 and passed |
| 4 | Relative brightness: keep regions at least 85% as bright as the brightest kept region | Measured: eggs 232 to 238, wall 141 to 150, reflections 137 | Clean: 2 eggs when visible, 1 when the hand covers one, 0 on title cards |

Photo results are unchanged (dev 0.524); the video settings live in `VIDEO_SETTINGS` in `scripts/classical.py`.

## Tracker (Task 4) iterations, `scripts/track.py`

Detection: v8 with `VIDEO_SETTINGS` per frame at 640 wide. Matching: Hungarian assignment on centre
distance (gate 60 px). The video holds two eggs, so the right answer is two IDs.

| # | Settings | IDs produced | What went wrong |
|---|---|---|---|
| 1 | memory 0.5 s, 3 frames to confirm a new ID | 7 | The hand carries an egg across the table while hiding it; it reappears far away after 3 s and gets a new ID. A wrist (9 frames) and the fade to the title card (3 frames) each got an ID. Hidden tracks lingered half a second into the title card |
| 2 | memory 3 s, 12 frames to confirm; a leftover detection is offered to a hidden track with no distance limit (eggs do not appear from nowhere); scene cut (mean frame change > 40) drops all tracks | 3 | One egg hidden by the spinning hand for 3.7 s, past the 3 s memory |
| 3 | memory 5 s | 2 | Both IDs span the egg segment, but during the iris fade at 41.7 s the brightening wall passed the shape rules and the re-link rule gave it egg 1's ID; both tracks lingered onto the title card because the fade is gradual |
| 4 | re-link only to a detection between half and double the egg's area; scene change also checked against the frame 0.5 s earlier so fades count | **2** | Egg 1 and egg 2: frames 243 to 998 (10.1 s to 41.6 s), 1281 detections. Nothing on title cards or during the fade |

Caveat: the re-link rule assumes a detection appearing while an egg is missing is that egg. With two
identical eggs this cannot be verified by appearance; if both were hidden and swapped, the IDs would
swap too. It did not happen in this clip as far as the sampled frames show.
