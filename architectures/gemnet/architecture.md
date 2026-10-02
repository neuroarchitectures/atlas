# Architecture: GemNet

## Motivation

Geometric Message Passing Neural Network with directional message passing for molecular property prediction.

## Core Idea

Geometric Message Passing Neural Network with directional message passing for molecular property prediction.

## Architecture

### Overview

![gemnet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Molecular Graph | `input` | shape=[100] |
| 2 | Atom Embedding | `embedding` | vocabSize=100, dim=128 |
| 3 | Interaction Blocks (6x) | `gnn` | hiddenSize=128 |
| 4 | Property Head | `linear` | outFeatures=1 |
| 5 | Molecular Property | `output` | — |

</details>

### Components

1. **Molecular Graph** — input layer. 2. **Atom Embedding** — embedding layer. 2. **Interaction Blocks (6x)** — gnn layer. 2. **Property Head** — linear layer. 2. **Molecular Property** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Geometric Message Passing Neural Network with directional message passing for molecular property prediction.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
