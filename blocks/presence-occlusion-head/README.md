# Presence / Occlusion Head

## Design Philosophy

Promptable segmentation silently fails when the object is absent or hidden — the model returns a confident garbage mask. Add an explicit binary head scoring *whether the target is visible at all*, trained with positive and negative (absent/occluded) examples, so the system can answer "not present" instead of hallucinating.

## Functionality

- Linear head on decoder output (SAM 2: per-object occlusion score, gates masklet continuation and memory writes; SAM 3: per-concept presence score before instance masks).
- Negative training data: frames without the object / occluded instances.

## Used By

| Model | Role |
|-------|------|
| SAM 2 | Occlusion head gating masklet continuation |
| SAM 3 | Presence head enabling zero-instance answers for concept prompts |

## Features

- **Explicit negative space** — the head that makes "find all instances of X" answerable with "none".
- **Drives control flow** — occlusion score decides memory updates, not just confidence display.

## Evolution

- **Predecessor**: implicit confidence scores (predicted IoU in SAM).
- **Related**: multi-mask-iou-head — quality scoring rather than existence scoring.
