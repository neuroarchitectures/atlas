# SimCC Coordinate Classification Head

## Design Philosophy

Heatmap pose estimation wastes compute upsampling with deconvolutions and quantizes coordinates to heatmap resolution. SimCC reformulates pose as *two independent 1D classification problems*: bin the x and y axes, and classify each keypoint into its horizontal and vertical bin — no upsampling, sub-pixel bins.

## Functionality

- Pooled features → two linear branches producing per-keypoint logits over S_x × S_y bins.
- Normalized coordinate via UDP transform for training targets and decoding; prediction = (argmax_x, argmax_y).

## Used By

| Model | Role |
|-------|------|
| RTMPose | SimCC head over CSPNeXt features; top-down and one-shot variants |

## Features

- **No deconvolution stack** — decoupled from feature resolution; bin granularity is free.
- **Sub-pixel precision** from binning, not interpolation.

## Evolution

- **Predecessor**: heatmap regression (Hourglass, HRNet heads).
- **Successor**: distribution-based regression (RTMPose 2.x); DiffPose diffusion formulations.
