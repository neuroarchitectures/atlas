# Keypoint Heatmap Head

## Design Philosophy

Represent each body keypoint as a 2D Gaussian peak in its own heatmap channel. Training reduces to pixel-wise regression against the Gaussian target; inference to argmax (plus optional offset refinement). One heatmap per keypoint *type* keeps channels small and semantically clean.

## Functionality

- Decoder: deconv stack → K heatmap channels; target = Gaussian centered at GT joint with radius tied to box scale.
- Loss: MSE per channel; decode = argmax (+ quarter-pixel offset refinement).

## Used By

| Model | Role |
|-------|------|
| ViTPose | Plain non-hierarchical ViT + minimal deconv decoder, top-down on person crops |

## Features

- **Simple, stable supervision** — no binning, no regression outliers.
- **Model-agnostic head** — the classic pairing with transformer backbones.

## Evolution

- **Predecessor**: Hourglass, CPN heatmap heads.
- **Successor**: simcc-head (RTMPose) — classification replaces heatmap regression.
