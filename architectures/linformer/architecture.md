# Architecture: Linformer

## Motivation

Standard attention computes L x L attention matrix. Linformer projects K and V from L to k dimensions (k << L), making attention O(L*k) instead of O(L^2).

## Core Idea

Add two projection matrices E and F that project K and V from [L, d] to [k, d]. The attention becomes Q (E K)^T (F V)^T, which is O(L*k*d) instead of O(L^2*d).

## Architecture

### Overview

![linformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Tokens | `input` | shape: [4096, 512] |
| 2 | LayerNorm | `layerNorm` |  |
| 3 | QKV | `linear` | outFeatures: 1536 |
| 4 | Project K (L->k) | `linear` | outFeatures: 256, inFeatures: 4096 |
| 4 | Project V (L->k) | `linear` | outFeatures: 256, inFeatures: 4096 |
| 5 | Low-Rank Attention | `attention` | Q @ K_proj^T @ V_proj |
| 6 | Output | `output` |  |

</details>

### Components

1. **K and V projection** — Two learned projection matrices E_K and E_V project K and V from [L, d] to [k, d], where k << L (e.g., k=256 for L=4096). 2. **Low-rank attention** — Attention(Q, E_K K, E_V V) = softmax(Q (E_K K)^T) E_V V. This is O(L*k*d) instead of O(L^2*d). 3. **Empirical low-rank** — The authors showed that the attention matrix is empirically low-rank, justifying the projection. 4. **Shared projections** — Projections can be shared across layers and heads for further efficiency.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Add two projection matrices E and F that project K and V from [L, d] to [k, d]. The attention becomes Q (E K)^T (F V)^T, which is O(L*k*d) instead of O(L^2*d).
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Transformer. Successor: Nyströmformer, FlashAttention.

## References

- Wang et al. 2020
