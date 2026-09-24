# Windowed Local Scan

## Design Philosophy

Global scan orders force distant pixels to be adjacent in the sequence, hurting locality-sensitive vision tasks. LocalMamba divides the image into windows and scans *within* them — adjacent tokens stay adjacent — with the scan direction searched independently per layer.

## Functionality

- Partition into windows (Swin-style) → directional scan (H/V/diagonal, chosen per layer by search) → selective SSM within each window → merge.
- Plain and hierarchical variants; direction search is a lightweight NAS over a small space.

## Used By

| Model | Role |
|-------|------|
| LocalMamba | Per-layer windowed scan feeding S6 blocks, plain + hierarchical backbones |

## Features

- **Locality-preserving traversal** — the fix for SS2D's global-order artifacts.
- **Per-layer direction flexibility** at negligible search cost.

## Evolution

- **Predecessor**: ss2d-selective-scan (four global routes).
- **Related**: swin-shifted-window-attention — the attention-side windowing precedent.
