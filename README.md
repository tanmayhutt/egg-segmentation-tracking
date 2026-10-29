# Egg segmentation and tracking

Computer vision assignment for Nosh Robotics (Euphotic Labs). Four tasks on 110 photos of eggs
and one video:

1. **SAM masks.** Segment every egg with the Segment Anything Model to get reference masks.
2. **Classical segmenter.** Hand-written rules (no learning) that find eggs, tuned on a
   leak-free 4:1 dev/eval split, with every attempt logged.
3. **Evaluation.** IoU, precision and recall of the classical masks against SAM's on eval.
4. **Tracking.** Follow each egg separately through a video.

The report is `docs/report.pdf`. The full trial-by-trial record is `docs/experiment_log.md`.

## Results in one table

| | Dev (88 photos) | Eval (22 photos) |
|---|---|---|
| Classical v1 baseline, mean IoU | 0.371 | 0.188 |
| Classical v8 final, mean IoU | 0.524 | 0.472 |
| v8 precision / recall | 0.61 / 0.68 | 0.54 / 0.64 |

Tracking: 2 IDs for the 2 eggs across the whole 32 s egg segment of the video.
Tracked video: https://drive.google.com/file/d/1DC3uYRb500AILF8IBrBE3OaTCb-4HBh4/view?usp=sharing

## Setup

Tested on macOS (Apple M2, PyTorch on the MPS backend) with Python 3.12.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Without `uv`: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

Data and weights are not in the repo:

- Dataset: download the Roboflow export (link in the assignment) and unzip so that photos sit at
  `data/egg/{train,valid,test}/egg/*.jpg`.
- SAM weights: `weights/sam_vit_b_01ec64.pth` from the Segment Anything release
  (`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`, 375 MB).
- Video: `data/video/eggs.mp4`, downloaded with `yt-dlp` from the YouTube link in the assignment
  (720p stream, `-f 136`).

## Run order

All scripts run from the repo root with `.venv/bin/python`.

| Step | Command | Output | Time |
|---|---|---|---|
| Look at the data | `scripts/contact_sheet.py` | `outputs/contact_sheet.jpg` | seconds |
| Task 1a, SAM candidates (slow) | `scripts/sam_candidates.py` | `outputs/sam_candidates/*.npz` | ~35 min |
| Task 1b, choose egg masks | `scripts/sam_select.py` | `outputs/sam_masks/*.png`, `summary.csv`, `outputs/sam_review.jpg` | ~1 min |
| Task 2a, split | `scripts/make_split.py` | `config/split.json`, `outputs/split_review.jpg` | seconds |
| Task 2b, classical on dev | `scripts/classical.py v8 dev` | score and `outputs/classical/v8_dev.jpg` | ~1 min |
| Task 3, eval | `scripts/evaluate.py v8 eval` | `outputs/eval/v8_eval.csv`, `.jpg` | seconds |
| Task 4, tracking | `scripts/track.py` | `outputs/video/tracked.mp4`, `tracks.csv` | ~40 s |
| Report PDF | `scripts/build_report.py` | `docs/report.pdf` | seconds |

`scripts/classical.py` accepts any version `v1` to `v8`, so every row of the experiment log can
be re-run. `scripts/sam_explore.py 1 17 42` draws every SAM candidate on chosen photos.

## Layout

```
scripts/      all code (common.py, sam_model.py, metrics.py are shared helpers)
config/       hand-made inputs: SAM clicks for 14 photos, scene groups, the frozen split
docs/         report (md + pdf), experiment log, study notes
outputs/      results and review sheets (SAM candidates and the video are gitignored)
```

## Manual inputs, stated plainly

- 14 of 110 photos needed a hand-placed click for SAM, listed with reasons in
  `config/sam_prompts.json`. The other 96 are fully automatic.
- The 24 scene groups for the split were made by eye (`config/scene_groups.json`) after three
  automatic grouping methods failed; see the experiment log.

## AI tools

An AI assistant (Claude) was used as a coding aid: drafting script code, helping debug, and
explaining methods on request. The approach, the decisions at each step, the thresholds (all chosen
from numbers measured on the dev photos), the checking of every result by eye, and the experiment
log in `docs/experiment_log.md` are my own work. No egg masks were produced by a language model.
