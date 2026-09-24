# Multi-Scale Deformable Attention (MSDeformAttn)

## Design Philosophy

Full attention over an image is O(HW); deformable attention makes it linear: each query predicts a small set of *sampling offsets* and attends to K sampled points per head across L feature levels, with bilinear interpolation. Multi-scale reference points give the mechanism scale awareness for free.

## Functionality

- Per query with 2D reference point p: predict offsets Δp_mk and weights a_mk; output `Σ_m Σ_k a_mk · x_m(p + Δp_mk)`.
- Typically K=4 points per level per head; levels share predicted offsets scaled per level.

## Used By

| Model | Role |
|-------|------|
| Deformable DETR | Encoder + decoder attention over FPN levels |
| DINO | Deformable attention in the mixed query selection decoder |
| D-FINE | Decoder attention for the distribution refinement head |
| Mask2Former | Pixel-decoder deformable attention over multi-scale pyramid |
| BEVFormer | Spatial cross-attention: projected 3D reference points per camera with visibility masking |

## Features

- **Linear in pixels** — the workhorse that made DETR-style detectors practical.
- **Learned where to look** — data-dependent sampling replaces global attention.

## Evolution

- **Predecessor**: deformable convolution (DCN-v2) — the conv-side ancestor.
- **Successor**: RT-DETR's real-time design replaces it with standard attention + selective queries; sparsebev scale-adaptive attention in 3D.
