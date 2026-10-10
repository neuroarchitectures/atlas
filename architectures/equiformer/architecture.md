# Architecture: Equiformer

## Motivation

Equivariant GNNs (NequIP, PaiNN) use message passing but lack global attention. Transformer attention could improve long-range interactions.

## Core Idea

Replace message passing with equivariant self-attention using spherical harmonics, combining the expressiveness of Transformers with SE(3)-equivariance.

## Architecture

### Overview

![equiformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Atomic Positions | `input` | shape: [20, 3] |
| 2 | Atom Embedding | `embedding` | dim: 128 |
| 3 | Equivariant Attention | `equivariant-attention` |  |
| 4 | Equivariant Attention | `equivariant-attention` |  |
| 5 | Equivariant Attention | `equivariant-attention` |  |
| 6 | Global Pooling | `pooling` |  |
| 7 | Property Head | `linear` | outFeatures: 1 |
| 8 | Molecular Property | `output` |  |

</details>

### Components

1. **Atom embedding** — Per-element embedding. 2. **Equivariant attention** — Self-attention with irreducible representations (spherical harmonics up to L=2). 3. **Equivariant FFN** — Tensor-product-based feedforward. 4. **Global pooling** — Sum over atoms. 5. **Property head** — Scalar output.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Replace message passing with equivariant self-attention using spherical harmonics, combining the expressiveness of Trans
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: NequIP, PaiNN. Successor: EquiformerV2.

## References

Liao & Smidt 2023
