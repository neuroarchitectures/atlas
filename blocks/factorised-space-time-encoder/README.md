# Factorized Space-Time Encoder

## Design Philosophy

Instead of dividing attention inside one stack (TimeSformer), split the *encoder* itself: a spatial transformer builds per-frame tokens, a temporal transformer models their evolution, with intermediate token connections — factorizing the space-time cost at the architecture level and letting each stack specialize.

## Functionality

- Variant: separate spatial and temporal transformer stacks, intermediate tokens passed between them (ViViT Model 2/3); tubelet-embedding tokenizes the input.
- Pretraining-friendly: initialize from image-pretrained models; regularization matters in the small-data video regime.

## Used By

| Model | Role |
|-------|------|
| ViViT | Factorized encoder variants for video classification |

## Features

- **Specialized stacks** — spatial semantics and temporal dynamics don't share weights.
- **Image-model initialization** — a practical pretraining advantage.

## Evolution

- **Predecessor**: two-stream CNNs (SlowFast); joint video ViT.
- **Related**: divided-space-time-attention — the in-block factorization; tube-masking supplies the pretraining objective.
