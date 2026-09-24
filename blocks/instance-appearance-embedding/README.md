# Instance Appearance Adapter

## Design Philosophy

Tracking-by-detection needs appearance embeddings robust across domains, but training them requires labels. MASA's adapter sits on a *frozen* detection/segmentation backbone and converts its features into universal instance appearance embeddings — supervised by dense correspondence mining from unlabeled images (same instance across views/frames = positive pairs), plus a distillation branch from a teacher.

## Functionality

- Frozen SAM/detector features → adapter (trained) → appearance embedding; cosine-similarity association.
- Correspondence supervision mined from unlabeled video/images; optional teacher distillation.

## Used By

| Model | Role |
|-------|------|
| MASA | Universal appearance representation for detect/segment-and-track adapters over SAM, DINOv2, DETR backbones |

## Features

- **Label-free training** — correspondence replaces tracked-video annotation.
- **Foundation-model reuse** — the adapter is small; the backbone stays frozen.

## Evolution

- **Predecessor**: learned ReID embeddings (Tracktor/ByteTrack era).
- **Related**: embedding-distillation (mobilesam) — adapter-over-frozen-encoder as the shared pattern.
