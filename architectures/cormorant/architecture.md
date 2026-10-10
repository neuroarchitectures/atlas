# Architecture: Cormorant

## Motivation

COvariant Molecular Neural Network with Learned Attention: SE(3)-covariant attention for molecular energy.

## Core Idea

COvariant Molecular Neural Network with Learned Attention: SE(3)-covariant attention for molecular energy.

## Architecture

### Overview

![cormorant architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Molecular Positions | `input` | shape=[100, 3] |
| 2 | Feature Embedding | `embedding` | vocabSize=100, dim=64 |
| 3 | Covariant Attention Layers (4x) | `gnn` | hiddenSize=64 |
| 4 | Global Pooling | `linear` | outFeatures=1 |
| 5 | Molecular Energy | `output` | — |

</details>

### Components

1. **Molecular Positions** — input layer. 2. **Feature Embedding** — embedding layer. 2. **Covariant Attention Layers (4x)** — gnn layer. 2. **Global Pooling** — linear layer. 2. **Molecular Energy** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — COvariant Molecular Neural Network with Learned Attention: SE(3)-covariant attention for molecular energy.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
