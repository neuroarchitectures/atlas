# DPT Reassembly + Fusion Decoder

## Design Philosophy

Plain ViT features are single-resolution; dense prediction needs multi-scale detail. Reassemble each transformer stage's tokens back to a chosen spatial resolution (with projection to a common channel width), then progressively fuse and upsample the levels to full resolution.

## Functionality

- Per stage: de-tokenize → resample (conv-transpose / identity / strided conv / bilinear) to target resolution → 1×1 projection.
- Progressive fusion: iteratively combine current fusion output with the next level via 3×3 convs and upsampling to full resolution.

## Used By

| Model | Role |
|-------|------|
| Depth Anything V2 | Reassembly + fusion over DINOv2 S/B/L stages for affine-invariant depth |
| MoGe | DPT-style decoder producing pointmap/mask/scale-focal heads |

## Features

- **Resolution per stage is a free choice** — early stages keep high res, late stages keep semantics.
- **Head-agnostic** — attaches to any ViT-style multi-stage encoder.

## Evolution

- **Predecessor**: DPT (Dense Prediction Transformers, 2021); SETR naive upsampling.
- **Related**: all-mlp-decoder (SegFormer), unet-encoder-decoder (convolutional counterpart).
