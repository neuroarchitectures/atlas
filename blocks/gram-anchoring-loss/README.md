# Gram Anchoring Loss

## Design Philosophy

Continued pretraining on uncurated data erases dense-feature quality learned from curated data. DINOv3's Gram anchoring regresses the student's patch-feature *Gram matrix* toward a frozen teacher's (a small model trained once on curated data) — preserving the geometry of feature similarity without distilling every activation.

## Functionality

- `L = ‖G_student(patches) − G_teacher(patches)‖` with G = normalized feature Gram matrix over patch pairs.
- Added to the DINO/iBOT/ICT self-supervised losses during large-scale uncurated training (ViT-S…7B).

## Features

- **Geometry-level distillation** — one matrix captures the similarity structure; cheaper than token-level distillation.
- **Curated→uncurated bridge** — the mechanism behind DINOv3's dense-feature claims.

## Used By

| Model | Role |
|-------|------|
| DINOv3 | Gram anchoring across the encoder family |

## Evolution

- **Predecessor**: knowledge distillation (logits/features); DINOv2's curated-data recipe.
- **Related**: embedding-distillation (MSE-level); go-lsd-self-distillation (cross-layer).
