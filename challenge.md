---
company: Nosh Robotics (Euphotic Labs)
role: Computer Vision Intern
sent: 2026-10-28 19:20 IST by Sudeep Gupta via Wellfound
deadline: 2026-11-01 23:59 IST
brief: https://drive.google.com/file/d/1_2_S2x3kiZKvrA0mOVVsSJ1OSJrVvY5m/view
submit_form: https://forms.gle/yuPDTNgS3EjUpgzS7
---

# Nosh CV Intern task

Transcribed from "CV Intern Task Wellfound.pdf" (2 pages). Links were extracted from the PDF itself.

## Links

- Dataset (Roboflow): https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1
- Dataset (alternative zip): https://drive.google.com/file/d/1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv/view?usp=sharing
- Suggested model: https://github.com/facebookresearch/segment-anything
- Tracking video: https://www.youtube.com/watch?v=Y4JyV94y2_A

## Task 1. Segmentation using a pretrained model

Use a pretrained segmentation model (for example SAM) to identify and segment the egg(s) in all
images in the dataset.

## Task 2. Segmentation using image processing

- Split the dataset into a dev split and an eval split, 4:1. Reduce data leakage between them.
- Build a segmentation algorithm with classical image processing that segments eggs, matching
  Task 1's quality as closely as possible.
- Experiment log in the report covering every trial, including failures. For each: what changed,
  why, parameters, dev-set results, why kept or discarded. Show how the trials led to the final
  algorithm.

## Task 3. Evaluation and analysis

Treat Task 1's masks as ground truth and Task 2's outputs as predictions. Compute relevant
segmentation metrics and write a detailed analysis of performance gaps and improvements. Both
quantitative and qualitative: visualisations of wrongly segmented samples and plots showing what
went wrong, with a brief but detailed write-up.

## Task 4. Video tracking

Run the final algorithm on the YouTube video and track individual eggs separately. The
segmentation algorithm may be modified and/or a tracking algorithm added.

## AI tools

Allowed. Briefly disclose where they were used. "You're not expected to outsource your thinking to
these tools and are expected to clearly understand and explain the underlying methods." The initial
interview is a walkthrough of the algorithm implementation, making modifications the interviewer
suggests, live.

## Submission

- Public GitHub repo (GitHub or Bitbucket only) with a PDF report covering all tasks, committed to
  the repo.
- The PDF links to a publicly accessible Google Drive link holding the video tracking output.
- Setup and run instructions in the repo.
- Partial submissions are acceptable.
- The PDF says 7 days from receipt; the email says 2026-11-01 23:59 IST. Use the email's date.
- Then fill https://forms.gle/yuPDTNgS3EjUpgzS7
