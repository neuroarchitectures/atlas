# Architecture: TFN

## Motivation

Tensor Field Network: SE(3)-equivariant neural network using spherical harmonics and tensor products.

## Core Idea

Tensor Field Network: SE(3)-equivariant neural network using spherical harmonics and tensor products.

## Architecture

### Overview

![tfn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Molecular Positions | `input` | shape=[100, 3] |
| 2 | Feature Embedding | `embedding` | vocabSize=100, dim=64 |
| 3 | Tensor Product Convolution | `gnn` | hiddenSize=64 |
| 4 | Global Pooling | `linear` | outFeatures=128 |
| 5 | Molecular Property | `output` | — |

</details>

### Components

1. **Molecular Positions** — input layer. 2. **Feature Embedding** — embedding layer. 2. **Tensor Product Convolution** — gnn layer. 2. **Global Pooling** — linear layer. 2. **Molecular Property** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Tensor Field Network: SE(3)-equivariant neural network using spherical harmonics and tensor products.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
