# Architecture: AutoInt

## Motivation

Automatic feature interaction learning via multi-head self-attention over feature embeddings.

## Core Idea

Automatic feature interaction learning via multi-head self-attention over feature embeddings.

## Architecture

### Overview

![autoint architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Sparse Features | `input` | shape=[100] |
| 2 | Embedding Layer | `embedding` | vocabSize=1000, dim=16 |
| 3 | Self-Attention Layer 1 | `transformer-encoder` | numLayers=1, hiddenSize=16, numHeads=2 |
| 4 | Self-Attention Layer 2 | `transformer-encoder` | numLayers=1, hiddenSize=16, numHeads=2 |
| 5 | Self-Attention Layer 3 | `transformer-encoder` | numLayers=1, hiddenSize=16, numHeads=2 |
| 6 | Prediction | `output` | — |

</details>

### Components

1. **Sparse Features** — input layer. 2. **Embedding Layer** — embedding layer. 2. **Self-Attention Layer 1** — transformer-encoder layer. 2. **Self-Attention Layer 2** — transformer-encoder layer. 2. **Self-Attention Layer 3** — transformer-encoder layer. 2. **Prediction** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Automatic feature interaction learning via multi-head self-attention over feature embeddings.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
