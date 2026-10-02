# Architecture: BEiT

## Motivation

Self-supervised pre-training (MAE) reconstructs pixels, but pixel-level reconstruction is noisy and low-level.

## Core Idea

Use masked image modeling with discrete VQ-VAE tokens as prediction targets, similar to BERT's masked language modeling, for cleaner self-supervised pre-training.

## Architecture

### Overview

![beit architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | Patch Embedding | `patchify` | patchSize: 16 |
| 3 | Positional Encoding | `positional-encoding` | learned |
| 4 | Transformer Encoder | `transformer` | 12 layers |
| 5 | LayerNorm | `layernorm` |  |
| 6 | Classification Head | `linear` | outFeatures: 1000 |
| 7 | Class Logits | `output` |  |

</details>

### Components

1. **Patch embedding** — 16x16 patches to 768-dim. 2. **Masking** — Randomly mask ~40% of patches. 3. **Transformer encoder** — 12-layer ViT. 4. **Prediction** — Predict VQ-VAE tokens for masked patches. 5. **Fine-tuning** — Add classification head for downstream tasks.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use masked image modeling with discrete VQ-VAE tokens as prediction targets, similar to BERT's masked language modeling,
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ViT, MAE. Successor: BEiT-2, BEiT-3.

## References

Bao et al. 2021
