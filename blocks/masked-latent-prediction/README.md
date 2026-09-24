# Masked Latent Prediction (JEPA)

## Design Philosophy

Reconstructing pixels forces models to model noise and texture they don't need. JEPA instead predicts the *representations* of masked regions: an online encoder processes visible context, a predictor maps it to predicted target features, and the loss compares predictions against a stop-gradient EMA target encoder's features — no pixels, no hand-crafted augmentations, no negatives.

## Functionality

- Context encoder (ViT) → predictor (ViT) → predicted target-block features; targets from EMA target encoder (stop-grad).
- Masking scale carries the semantics: large context blocks / distributed targets (I-JEPA); spatio-temporal tubes for video (V-JEPA); smooth-L1/L2 latent loss.

## Used By

| Model | Role |
|-------|------|
| I-JEPA | Image JEPA: large-scale targets + spatially distributed context |
| V-JEPA / V-JEPA-2 | Video JEPA over tube-masked latents (1M+ hours for V2) |
| V-JEPA 2-AC | Base objective; action-conditioned dynamics trained on top of frozen latents |

## Features

- **Semantic without pixels** — prediction difficulty lies in abstraction, not texture.
- **EMA stop-grad targets** — the same self-distillation trick as BYOL/DINO, inside a masked-prediction frame.

## Evolution

- **Predecessor**: masked-autoencoder (pixel targets), BYOL/DINO (EMA teachers).
- **Successor**: V-JEPA 2-AC adds action conditioning → latent world models.
