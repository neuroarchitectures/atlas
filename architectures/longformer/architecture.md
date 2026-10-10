# Architecture: Longformer

## Motivation

Transformer with local + global sliding-window attention for long-document processing.

## Core Idea

Transformer with local + global sliding-window attention for long-document processing.

## Architecture

### Overview

![longformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape=[4096] |
| 2 | Token Embedding | `embedding` | vocabSize=50265, dim=768 |
| 3 | Longformer Encoder (sliding-window + global attention) | `transformer-encoder` | numLayers=12, hiddenSize=768, numHeads=12 |
| 4 | Pooler | `linear` | outFeatures=768 |
| 5 | Pooled Output | `output` | — |

</details>

### Components

1. **Token IDs** — input layer. 2. **Token Embedding** — embedding layer. 2. **Longformer Encoder (sliding-window + global attention)** — transformer-encoder layer. 2. **Pooler** — linear layer. 2. **Pooled Output** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Transformer with local + global sliding-window attention for long-document processing.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
