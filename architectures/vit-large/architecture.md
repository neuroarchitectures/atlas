# Architecture: ViT-Large

## Motivation

ViT-Base showed promise but larger models trained on larger datasets could achieve state-of-the-art performance.

## Core Idea

Scale up the Vision Transformer to 24 layers, 1024 hidden size, 16 heads, and 307M parameters, trained on large datasets (JFT-300M or ImageNet-21k).

## Architecture

### Overview

![vit-large architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | Patch Embedding | `patchify` | patchSize: 16 |
| 3 | Positional Encoding | `positional-encoding` | learned |
| 4 | Transformer Encoder | `transformer` | 24 layers |
| 5 | LayerNorm | `layernorm` |  |
| 6 | Classifier | `linear` | outFeatures: 1000 |
| 7 | Class Logits | `output` |  |

</details>

### Components

1. **Patch embedding** — 16x16 patches projected to 1024-dim. 2. **Learned positional encoding** — Adds spatial information. 3. **Transformer encoder** — 24 layers of multi-head self-attention and MLP. 4. **Classification head** — CLS token + linear.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Scale up the Vision Transformer to 24 layers, 1024 hidden size, 16 heads, and 307M parameters, trained on large datasets
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ViT-Base. Successor: ViT-Huge, DeiT, DINOv2.

## References

Dosovitskiy et al. 2021
