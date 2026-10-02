# Architecture: TorchMD-Net

## Motivation

Classical force fields require manual parameterization. GNN-based potentials (SchNet, NequIP) are limited by message passing.

## Core Idea

Use a Transformer architecture with attention over atomic neighborhoods, enabling efficient and expressive interatomic potentials for molecular dynamics.

## Architecture

### Overview

![torchmd-net architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Atomic Positions | `input` | shape: [20, 3] |
| 2 | Atom Embedding | `embedding` | dim: 128 |
| 3 | Attention | `attention` |  |
| 4 | Attention | `attention` |  |
| 5 | Attention | `attention` |  |
| 6 | Attention | `attention` |  |
| 7 | Global Pooling | `pooling` |  |
| 8 | Energy Head | `linear` | outFeatures: 1 |
| 9 | Total Energy | `output` |  |

</details>

### Components

1. **Atom embedding** — Per-element learnable embedding. 2. **Attention layers** — Multi-head self-attention over neighbors, conditioned on pairwise distances. 3. **Distance-dependent attention** — Attention weights modulated by radial basis function of distances. 4. **Global pooling** — Sum over atoms. 5. **Energy head** — Scalar prediction.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use a Transformer architecture with attention over atomic neighborhoods, enabling efficient and expressive interatomic p
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: SchNet, PhysNet. Successor: Equiformer.

## References

Thölke & De Fabritiis 2022
