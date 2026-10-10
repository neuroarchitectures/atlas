# Architecture: Performer

## Motivation

Transformer attention is O(L^2), limiting sequence length. Performer uses random feature approximations of softmax to achieve linear-time attention while preserving quality.

## Core Idea

Use FAVOR+ (Fast Attention via positive Orthogonal Random features) to decompose the softmax kernel into features: K(x,y) approx E[phi(x) . phi(y)], enabling (Q phi) (K phi)^T V instead of Q K^T V.

## Architecture

### Overview

![performer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Tokens | `input` | shape: [4096, 512] (long sequence) |
| 2 | LayerNorm | `layerNorm` |  |
| 3 | QKV Projection | `linear` | outFeatures: 1536 (3x512) |
| 4 | FAVOR+ Attention | `attention` | O(L) via random features |
| 5 | Residual | `add` |  |
| 6 | LayerNorm 2 | `layerNorm` |  |
| 7 | FFN | `linear` | outFeatures: 2048 |
| 8 | Output | `output` |  |

</details>

### Components

1. **FAVOR+ decomposition** — Approximates softmax(QK^T) using random features: phi(Q) phi(K)^T, where phi uses random Gaussian projections followed by exp/normalization. 2. **Associative multiplication** — Compute (phi(Q) (phi(K)^T V)) instead of ((Q K^T) V). The inner product is O(L d r) instead of O(L^2 d). 3. **Positive random features** — Uses positive random features (h+) for better approximation of softmax. 4. **Orthogonal random features** — Uses orthogonal random matrices for lower variance.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use FAVOR+ (Fast Attention via positive Orthogonal Random features) to decompose the softmax kernel into features: K(x,y) approx E[phi(x) . phi(y)], enabling (Q phi) (K phi)^T V instead of Q K^T V.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Linear Attention, Transformer. Successor: FlashAttention (exact, not approximate).

## References

- Choromanski et al. 2021
