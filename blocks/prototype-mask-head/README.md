# Prototype Mask Head (YOLACT-style)

## Design Philosophy

Predicting a full mask per instance is expensive; instead predict a small bank of *prototype masks* shared across the image, plus per-instance *coefficient vectors* — the mask is a linear combination. This turns instance segmentation into dynamic filtering, cheap enough to run inside a real-time detector.

## Functionality

- Prototype branch: k prototype masks from backbone features (prototypes are class-agnostic).
- Per anchor: k coefficients via an extra conv; mask = `σ(Σ coeff_i · proto_i)` cropped by the predicted box.

## Used By

| Model | Role |
|-------|------|
| FastSAM | YOLOv8-seg backbone + prototype/coefficient heads; prompt handling = post-hoc candidate selection (point/box/text) |

## Features

- **One shared mask computation** — instance masks cost only k coefficient predictions each.
- **Real-time compatible** — no RoIAlign, no transformer decoder.

## Evolution

- **Predecessor**: YOLACT/YOLACT++.
- **Successor**: transformer mask decoders (Mask2Former, SAM's two-way-transformer) trade the speed for quality.
