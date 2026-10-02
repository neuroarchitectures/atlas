# Architecture: DiffPool

## Motivation

Differentiable graph pooling: learned hierarchical node assignment for graph-level tasks.

## Core Idea

Differentiable graph pooling: learned hierarchical node assignment for graph-level tasks.

## Architecture

### Overview

![diffpool architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Graph Features | `input` | shape=[100, 100] |
| 2 | GNN Layer 1 | `gnn` | hiddenSize=64 |
| 3 | Differentiable Pool 1 | `gnn` | hiddenSize=64 |
| 4 | GNN Layer 2 | `gnn` | hiddenSize=64 |
| 5 | Differentiable Pool 2 | `gnn` | hiddenSize=64 |
| 6 | Global Readout | `linear` | outFeatures=128 |
| 7 | Graph Prediction | `output` | — |

</details>

### Components

1. **Graph Features** — input layer. 2. **GNN Layer 1** — gnn layer. 2. **Differentiable Pool 1** — gnn layer. 2. **GNN Layer 2** — gnn layer. 2. **Differentiable Pool 2** — gnn layer. 2. **Global Readout** — linear layer. 2. **Graph Prediction** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Differentiable graph pooling: learned hierarchical node assignment for graph-level tasks.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
