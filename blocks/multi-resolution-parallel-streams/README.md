# Multi-Resolution Parallel Streams (HRNet)

## Design Philosophy

Standard classification backbones treat high resolution as a *phase*: convolve down early, recover later. HRNet instead *maintains* a high-resolution stream through the whole network, adding lower-resolution streams per stage and repeatedly fusing all of them — so spatial precision is never lost to downsampling.

## Functionality

- Stage 1: single high-res stream. Stages 2–4: add 1/4, 1/8, 1/16 resolution streams (strided convs).
- After each stage, every stream exchanges information: downsample via strided 3×3 convs, upsample via bilinear + 1×1 conv; streams are concatenated for the head.

## Used By

| Model | Role |
|-------|------|
| HRNet (W32/W40/W48) | 4-stage, up to 4 parallel streams; multi-resolution concat head for pose/seg/detection |

## Features

- **Persistent high resolution** — keypoint heatmap accuracy without a large deconv decoder.
- **Repeated multi-scale fusion** — richer features than single-path FPN-style top-down recovery.

## Evolution

- **Predecessor**: encoder-decoder U-Nets, feature-pyramid-network (resolution recovered, not preserved).
- **Successor**: HRNetV2 (multi-resolution head); inspired later hybrid backbones (e.g., Compact HRNet variants).
