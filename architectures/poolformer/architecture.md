# Architecture: PoolFormer

## Motivation

Transformers achieve great results, but is attention the key? PoolFormer shows that replacing attention with simple average pooling achieves comparable performance, suggesting the overall architecture (MetaFormer) matters more than the specific token mixer.

## Core Idea

A MetaFormer architecture where the token mixer is replaced with a simple 3x3 average pooling operation. The rest of the block (LayerNorm, MLP, residuals) is identical to a ViT block. Stacked in 4 stages with hierarchical feature maps.

## Architecture

### Overview

![poolformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image Patches | `input` | patchified |
| 2 | Pooling | `identity` | 3x3 avg pool |
| 3 | LayerNorm + Residual | `identity` | residual connection |
| 4 | MLP | `linear` | 2-layer MLP |
| 5 | Features | `output` | hierarchical stages |

</details>

### Components

1. **Average pooling** — replaces self-attention as token mixer. 2. **MetaFormer framework** — the general architecture (residual + norm + mixer + norm + MLP) is what matters. 3. **Hierarchical structure** — 4 stages with downsampling like Swin. 4. **GroupNorm** — used instead of LayerNorm in some variants. 5. **Proof of concept** — shows attention is not strictly necessary for good performance.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A MetaFormer architecture where the token mixer is replaced with a simple 3x3 average pooling operation. The rest of the...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ViT, ResNet. Successor: ConvFormer, MetaFormer variants.

## References

- Yu et al. 2022
