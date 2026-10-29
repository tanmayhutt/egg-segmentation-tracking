# Study notes

Plain-language notes on every step, written to be explained out loud in the interview.

## Step 0. Setup and data

- Python 3.12 in a project folder `.venv`, because PyTorch does not support the system Python 3.15.
- PyTorch runs on the M2 GPU through Apple's MPS backend.
- Dataset: 110 photos, all 640x640, from a Roboflow classification export. Classification means
  one label per photo (`not_damaged`), so there are **no masks**. Task 1 has to create them.
- Roboflow **stretched** every photo to a square, so egg shapes are slightly distorted.
- Many photos are the same scene shot several times (keyboard #17/18/19/43, steel bowl
  #113/114/128/129). Roboflow's own split puts the same scene in train and test, so it leaks.

**Likely question:** Why not use Roboflow's split? It mixes near-identical photos across
train and test, which inflates the test score, and the brief asks for 4:1, not 70/20/10.

## Task 1. SAM masks

### How SAM works

1. **Image encoder** (slow, about 1 s): turns the photo into an embedding, a summary of what is where.
2. **Prompt encoder**: takes a click, a box or a rough mask that says which object you mean.
3. **Mask decoder** (fast): returns a mask for that object, with up to 3 guesses and a
   confidence score (predicted IoU) for each.

SAM was trained on 11 million images and 1 billion masks. It knows what object boundaries look
like, but it does not know object names. It cannot be asked for "egg".

### Automatic mode

With no clicks available, SAM's automatic mode clicks a 32x32 grid of points over the photo,
keeps masks that are confident and stable, and removes duplicates. It returns masks for
everything: eggs, hands, keys, shadows, bowls. On our samples, 7 masks for a plain photo,
93 for a patterned cloth.

### Choosing which masks are eggs

Measured on 8 sample photos (`scripts/sam_explore.py`). A mask counts as an egg if it is:

| Rule | Why |
|---|---|
| 0.8% to 60% of the photo | Drops specks and whole-background masks |
| Brightness 120+ | Eggs measured 137+. Shadows, keys and ash were darker |
| Saturation 45 or less | Eggs are nearly colourless |
| Solidity 0.95+ | Area / convex hull area. An egg has no dents; a hand does |
| Ellipse fill 0.96+ | Area / fitted ellipse area. Eggs 0.97+, empty egg-tray cups 0.90 to 0.95 |
| One piece | The largest connected piece holds 95%+ of the mask |
| At least 25% of the biggest egg | Drops small egg-like things, like earbud heads |
| Not mostly inside another egg | SAM returns some eggs twice, and parts like a glare spot |

Photos where the rules fail get a hand-placed click on each egg in `config/sam_prompts.json`.
SAM turns each click into a mask. This is the only manual input, and it is listed per photo.

### Slow step and fast step are separate

`sam_candidates.py` runs SAM once and saves every mask (about 18 s per photo).
`sam_select.py` applies the rules in milliseconds. Changing a rule never means re-running SAM.

### Result

- 110 photos, 138 eggs (93 photos with 1 egg, 6 with 2, 11 with 3).
- **96 photos fully automatic. 14 needed clicks** (13%), listed with reasons in `config/sam_prompts.json`.
- Every photo was checked by eye on `outputs/sam_review.jpg`.

Why the automatic filter missed those 14, each a rule doing its job on a case it was not built for:

| Cause | Photos | Rule that rejected it |
|---|---|---|
| Leaves lying across the egg | #10, #11, #12 | Solidity (the leaves cut dents in the mask) |
| Dark, backlit photo | #20, #24 | Brightness |
| Egg cut off by the frame edge | #44, third egg in #144 | Ellipse fill |
| Rod lying across the egg | #117 | Solidity, and the egg is in two pieces |
| Grey egg on foil, misc | #39, #119, #160 | One of brightness, shape or relative size |
| Egg blurred by boiling water | #124, #135, #145 | No clean egg mask existed at all |

### Clicks, and SAM's three guesses

One click is ambiguous: it could mean a glare spot, the egg, or the bowl the egg is in. So SAM
returns 3 guesses at different sizes with a confidence score each, and we keep the most
confident. Two clicks on the same object (one each side of the rod in #117) remove the doubt,
so then SAM returns one guess.

For the boiling-water photos the most confident guess was the whole bowl (score 0.97). The
smallest guess was the egg (score 0.63 to 0.76, lower because bubbles blur its edge). For these 3
photos only we keep the smallest guess (`"pick": "smallest"`). Their egg boundaries are the least
certain in the dataset; say so in Task 3.

### Problems hit

- **MPS float64 error.** Apple's GPU has no 64-bit floats. SAM's automatic mode builds click
  coordinates as 64-bit numbers. Fixed in `scripts/sam_model.py` by converting them to 32-bit,
  which is far more precision than a pixel needs.

### Limits to state in the report

- Task 3 treats these masks as ground truth. They are SAM's opinion plus our filter, not true
  hand-drawn labels, so Task 3 measures agreement with SAM.

**Likely questions:**
- Why automatic mode and not boxes? The dataset has no boxes or clicks to give SAM.
- Why ViT-B and not ViT-H? ViT-H is 2.4 GB and slower. For 110 photos of a simple object, ViT-B
  gives clean egg edges, as the sample run shows.
- How would you remove the manual clicks? Use a detector for the prompts (for example
  Grounding DINO with the text "egg"), then give its boxes to SAM.

## AI tool use

- An AI assistant (Claude) was used as a coding aid for drafting code, debugging and explanations.
  The decisions, thresholds and checks are mine; every rule above was kept only after measuring it.

## Task 2a. The dev/eval split

**Why split at all.** The classical rules (thresholds, kernel sizes) get adjusted by looking at
results. If they are adjusted on the same photos they are scored on, the score is flattering
and says nothing about new photos. Dev is for adjusting. Eval is scored once, at the end.

**Why a plain random split leaks.** Many photos are the same scene shot again from a new angle
(keyboard #17/18/19/43, steel bowl #113/114/128/129). Put one in dev and one in eval and the
rules have effectively seen the eval photo already. That is data leakage.

**What was tried first, and failed.** Three automatic "fingerprints" (perceptual hash, colour
histogram, averaged SAM embedding) were tested on the known same-scene sets. None put them
closer together than unrelated photos. Perceptual hashing fingerprints the layout of light
and dark patches, which changes when the camera moves. Histograms and averaged embeddings see
"white egg on a grey surface" everywhere. Details in `docs/experiment_log.md`.

**What was done.** All 110 photos were grouped by eye into 24 scenes
(`config/scene_groups.json`). When unsure, photos were merged, because over-merging is safe
and under-merging leaks. `scripts/make_split.py` shuffles the groups with a fixed seed and deals
whole groups to eval until it holds one fifth of the photos. The seed (21, from 0 to 199) was
picked so eval's share of multi-egg photos and of click-fixed photos stays close to the whole set.

**Result.** Dev 88, eval 22, exactly 4:1. Eval scenes: striped towel (easy, white on white),
ash pile (hard, grey on grey), steel bowl with water including the three boiling-water photos
(hardest), brown marble, red lid. Eval has no hand photos and no beige-marble counter photos;
say so in the report.

**Likely questions:**
- Why by eye and not an algorithm? The algorithms were tested and failed on this data. With 110
  photos, looking is reliable and takes 15 minutes. A scene model like CLIP would likely work
  but adds a download for a one-off step.
- Why pick a seed? A random deal of 24 groups can land all easy scenes in eval. Trying 200 seeds
  and keeping the one whose eval mix matches the whole set is the same idea as stratified
  sampling, done at group level.
- Is the split fixed? Yes, it is written to `config/split.json`. Every later script reads it.

## Task 2b. The classical segmenter

"Classical" means no learning. Hand-written rules turn pixel values into an egg mask. Every number in
the rules was chosen by measuring on the 88 dev photos, and every attempt is a row in
`docs/experiment_log.md`. Final version: `v8` in `scripts/classical.py`, dev mean IoU 0.524.

### How the final pipeline works, step by step

1. **Colour space.** The photo goes from BGR to HSV: hue (which colour), saturation (how strong the
   colour), value (how bright). Measured on dev: eggs are bright (V 172 to 215) and nearly colourless
   (S 4 to 24). Hue is useless (a coin flip). So the two useful signals are high V and low S.
2. **Seed pixels.** Take the brightest 15% of pixels that are also colourless (S <= 80).
3. **Edge cut.** Canny finds edges (sharp brightness changes). Edge pixels, thickened to 5 px, are
   removed from the seeds, so an egg and the white bag it sits on start as separate pieces.
4. **Distance transform.** For each seed blob, measure how far every pixel is from the blob's edge.
   Keep only pixels at 70% or more of the blob's deepest point. A fat egg keeps its core; a thin
   bridge leaking into the surface is never deep, so it is cut off.
5. **Background seeds.** Pixels that are dark (darkest 30%) or strongly coloured (S > 60) are
   certainly not egg.
6. **Watershed.** Think of brightness as terrain and flood from every seed. Regions grow until they
   meet at edges. Each egg seed becomes one region.
7. **Trim leaks.** Each region is "opened" with a 31 px ellipse: erode, then dilate back. A neck
   joining the egg to a surface chunk is thinner than 31 px, so it breaks; the fat egg survives.
8. **Shape rules.** Keep regions that cover 2% to 60% of the photo, have solidity >= 0.85 (area over
   convex hull area, so no dents) and ellipse fill >= 0.9 (area over fitted-ellipse area). Drop
   regions under a quarter of the biggest kept one, and duplicates mostly inside a kept one.
9. **Fallback.** Every photo in this dataset has at least one egg. If nothing passes the rules, keep
   the single most egg-shaped region. Partial credit beats an empty mask.

### The story of the eight versions, in one paragraph each

- **v1, 0.371.** Grayscale, Otsu's automatic threshold, biggest bright blob. Otsu splits the
  brightness histogram into two groups. Fails when skin or a white surface is bigger than the egg.
- **v2, 0.425.** Add "colourless" (S <= 60) and morphology. Fixes hands in normal light. Steel trays,
  white labels and bags are still bright and colourless, and bigger than the egg.
- **v3, 0.283, worse.** Choose blobs by shape instead of size. Median zero. Diagnosis: in 51 of 88
  photos the egg was already merged into a bigger blob after thresholding, and no shape rule can
  split a merged blob. The lesson: measure where the failure is before changing the rule after it.
- **Side experiments.** Nine ways to threshold, scored by "does the egg survive as its own blob".
  Watershed with edge-cut seeds won (55 of 88). Local contrast with a 101 px window scored zero
  because the window was smaller than the egg.
- **v4, 0.350.** Watershed plus the v3 shape filter. Still bad: the selection step lost good regions.
- **v5, 0.465.** Smooth regions, minimum area 2%, relative-size rule. First non-zero median. Measured
  why: good egg regions have median area 9%, bad scraps 1%. Size, not shape, separated them.
- **v6, 0.450.** Distance-transform seeds. Eggs with a matching region 76 -> 86 of 109. Mean fell
  slightly because the selection rules were tuned for the old regions. A grid over thresholds
  found at most 0.477, so thresholds were not the lever.
- **v7, 0.451.** Coloured pixels as background seeds. Median 0.45 -> 0.62, but 42 photos still at 0.
- **v8, 0.524.** Trim leaks with a 31 px opening, and the fallback. Zero-score photos 37 -> 15.

### Where it works and where it fails (dev, per scene)

Works: marble counter (0.83, 13 photos, including all the touching-egg cases), patterned cloths
(0.83 to 0.84), keyboard (0.75), leaves (0.76). Fails: backlit hands (0.00, egg darker than the
background), white plastic bag (0.14), egg tray (0.10), foil (0.30), pink bowl with a white sticker
(0.36). The common thread: a white or very bright surface touching the egg with no visible edge
between them. Brightness, colour and edges are the only cues a classical method has, and all three
are absent there.

### Limits to state in the report

- Dev mean IoU 0.524 with precision 0.61 and recall 0.68: when it finds the egg it slightly
  over-segments, and in about a sixth of photos it finds nothing useful.
- The fallback uses a dataset fact (every photo has an egg). On photos without an egg it would
  return a wrong region every time.
- Eight versions and about 15 side experiments were all scored on dev. Eval was never looked at.
  The eval number in Task 3 is the honest one.

**Likely questions:**
- Why watershed and not just thresholding? Measured: thresholding alone left the egg as its own
  blob in at most 45 of 88 photos, watershed in 55, and with distance seeds 86 of 109 eggs.
- What is the distance transform doing? Finding the fat part of a blob. It is the standard way to
  separate touching round objects (coins, cells, eggs).
- Why did v3 get worse? The filter was fine; the input to it was already wrong. Debugging taught
  us to measure each stage separately.
- Would deep learning do better? Almost certainly, and SAM already shows it. The point of the
  task was to see how far rules go and to understand exactly why they stop.

## Task 3. Scoring on eval

Eval (22 photos) was run exactly once, with `scripts/evaluate.py`, after all tuning stopped.

| Version | Mean IoU | Precision | Recall |
|---|---|---|---|
| v1 baseline (Otsu, largest blob) | 0.188 | 0.206 | 0.605 |
| v8 final | **0.472** | 0.544 | 0.635 |
| v8 on dev, for comparison | 0.524 | 0.605 | 0.681 |

**How to read it.** Dev to eval dropped by 0.05. A small drop is expected: rules were tuned on dev, so
dev is slightly flattering. A large drop would mean over-tuning, a rise would be luck. The baseline
scored 0.19 on the same eval photos, so the tuning gained about 0.28 IoU on photos it never saw.

**What precision and recall say.** Precision 0.54: about half of what we call egg is not egg,
mostly extra bright regions (towel patches, steel rims, ash). Recall 0.64: we find about two thirds
of the egg pixels; the missing third is mainly the shaded side of each egg, which falls below the
brightness cut.

**Per scene.** Steel bowl with clear water 0.87 to 0.98. Boiling water 0.00 to 0.69. Ash pile 0.25,
with three photos at 0 where the egg is coated in ash. Towel 0.36, white on white.

**The honest limit.** The eval set has no hand photos and no beige-marble photos (the split put
them all in dev), and it does have the hardest scenes (boiling water, ash). So 0.47 is probably a
slightly pessimistic estimate. We report it as measured.

**Likely questions:**
- Why is eval lower than dev? Mild over-tuning, plus eval happens to hold the two hardest scenes.
- Is 0.47 good? For a rules-only method on photos with white-on-white and ash-on-ash scenes, it is
  realistic. SAM, a learned model, gets clean masks on 96 of 110 photos without any tuning. That
  gap is the point of the comparison.
- What would you change first? Fix the shaded-side recall with a second, lower brightness threshold
  applied only inside the fitted ellipse of each found egg. It would not help the zero-score photos.

## Task 4. The video

`data/video/eggs.mp4`: 60 s, 24 fps, 1440 frames, 1280x720. A science clip: two white eggs (one raw,
one boiled) on a dark brown table. A hand enters, spins each egg, and leaves. Title cards with no eggs
fill 0 to 10 s and 42 to 60 s. So about 32 s (770 frames) contain eggs, and the hard moments are
when the hand covers an egg and when an egg spins (motion blur, moving highlight).

### How the tracker works

1. **Detect** with v8 on each frame, resized to 640 wide (the scale the rules were tuned at), with
   four video settings: minimum area 0.4% instead of 2% (eggs are small in the frame), ellipse fill
   capped at 1.1 (a rounded rectangle over-fills its ellipse), keep only regions at least 85% as
   bright as the brightest one (the wall and table reflections are much darker than the eggs), and
   no fallback (title cards must return nothing). Photo results are unchanged.
2. **Match** each detection to an existing track by distance between centres, using the Hungarian
   algorithm (`scipy.optimize.linear_sum_assignment`). It finds the pairing with the smallest total
   distance, so two eggs never both grab the same detection. Matches over 60 px are refused.
3. **Hidden eggs.** A track with no match keeps its last position for up to 5 s (the hand hides a
   spinning egg for up to 3.7 s). It is drawn as a small circle marked "hidden".
4. **Re-link.** A detection that matches nothing nearby is offered to a hidden track with no distance
   limit, because eggs do not appear from nowhere: a new egg while one is missing is that egg,
   carried by the hand. The detection must be between half and double the egg's last area (same
   egg, same size). Only then does a detection start a new track.
5. **Birth.** A new track needs 12 consecutive frames (0.5 s) before it gets an ID. This removed a
   wrist and a fade that each passed the shape rules for a few frames.
6. **Scene change.** If the frame differs a lot (mean pixel change over 40) from the previous frame
   or from the frame half a second earlier, all tracks are dropped. The second check catches fades,
   which the first misses.

Result: 2 IDs for 2 eggs across the whole 32 s egg segment (10.1 s to 41.6 s), 1281 detections out
of about 760 frames that hold eggs (two eggs per frame when both are visible). Output in `outputs/video/`.

**Likely questions:**
- Why Hungarian and not nearest neighbour? With two eggs close together, greedy nearest neighbour
  can assign both tracks to one detection. Hungarian solves the whole assignment at once.
- What breaks it? Two eggs hidden at the same time and swapped. The eggs look identical, so no
  appearance cue could fix that. A Kalman filter would predict motion but eggs here move only when
  the hand moves them, which is unpredictable.
- Why not run SAM per frame? About 18 s per frame on this machine, 7 hours for the clip. The
  classical method runs the whole clip in 40 s.
