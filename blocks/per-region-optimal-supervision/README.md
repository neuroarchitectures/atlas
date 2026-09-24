# Per-Region Optimal Supervision

## Design Philosophy

Depth annotations are reliable in some image regions and garbage in others (sky, transparent, reflective). Instead of weighting the loss, MoGe's training-only module computes the *best per-region scale/shift alignment* of the prediction — with gradient masking so unreliable regions never backprop — before the pointmap regression loss.

## Functionality

- Partition prediction into regions; per region solve optimal affine (scale/shift) alignment to GT.
- Gradient masking zeroes unreliable-region gradients; module exists only in training, stripped at inference.

## Used By

| Model | Role |
|-------|------|
| MoGe | Supervision for affine-invariant pointmap regression |

## Features

- **Region-adaptive alignment** — finer than global scale-shift-invariant-loss.
- **Rejection by masking** — bad labels can't corrupt training.

## Evolution

- **Predecessor**: global scale-shift-invariant-loss alignment.
- **Related**: confidence-weighted-loss — predicted confidence as the alternative reliability signal.
