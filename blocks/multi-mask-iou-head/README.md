# Ambiguity-Aware Multi-Mask Output

## Design Philosophy

A single point prompt is inherently ambiguous — it might mean the whole object, a part, or a subpart. Instead of forcing one answer, emit several candidate masks (whole/part/subpart), each scored by a predicted-IoU head; the user (or downstream logic) resolves ambiguity by choice at prompt time.

## Functionality

- Mask decoder produces K masks per prompt (SAM: K=3) via K mask tokens.
- IoU prediction head (small MLP on decoder output + mask tokens) scores each mask; highest-confidence mask returned by default.

## Used By

| Model | Role |
|-------|------|
| SAM | 3 masks + predicted IoU per prompt |

## Features

- **Embraces ambiguity** — the defining UX decision of promptable segmentation.
- **Predicted IoU doubles as quality control** for downstream filtering.

## Evolution

- **Predecessor**: single-mask conditional segmentation.
- **Related**: presence-occlusion-head — a different explicit scoring head (existence, not quality).
