# Architecture: ChebNet

## Motivation

Spectral graph convolutions require eigendecomposition of the graph Laplacian, which is expensive. ChebNet approximates spectral filters with Chebyshev polynomials, enabling efficient localized graph convolutions.

## Core Idea

Graph convolution is approximated by a truncated Chebyshev polynomial expansion (order K) of the graph Laplacian. This captures K-hop neighborhood information without computing eigenvectors. The filter is strictly localized in space.

## Architecture

### Overview

![chebnet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Node Features | `input` | N x F |
| 2 | Chebyshev Conv | `conv1d` | K=3, order-3 polynomial |
| 3 | Chebyshev Conv 2 | `conv1d` | K=3 |
| 4 | Readout | `linear` | F -> num_classes |
| 5 | Node Classification | `output` | per-node labels |

</details>

### Components

1. **Chebyshev polynomial** — T_k(L̃) approximates spectral filter. 2. **K-hop localization** — order K captures K-hop neighborhood. 3. **No eigendecomposition** — avoids expensive Laplacian eigenvector computation. 4. **Recursive computation** — T_k computed recursively from T_{k-1} and T_{k-2}. 5. **Graph Laplacian** — normalized: L̃ = 2L/λ_max - I.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Graph convolution is approximated by a truncated Chebyshev polynomial expansion (order K) of the graph Laplacian. This c...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Spectral GCN (Bruna 2014). Successor: GCN (Kipf & Welling, first-order Chebyshev).

## References

- Defferrard et al. 2016
