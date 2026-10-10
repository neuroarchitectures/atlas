# Architecture: DistilBERT

## Motivation

Distilled BERT: 6-layer transformer encoder trained via knowledge distillation from BERT-base.

## Core Idea

Distilled BERT: 6-layer transformer encoder trained via knowledge distillation from BERT-base.

## Architecture

### Overview

![distilbert architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape=[512] |
| 2 | Token Embedding | `embedding` | vocabSize=30522, dim=768 |
| 3 | Transformer Encoder (6 layers) | `transformer-encoder` | numLayers=6, hiddenSize=768, numHeads=12 |
| 4 | Pooler | `linear` | outFeatures=768 |
| 5 | Pooled Output | `output` | — |

</details>

### Components

1. **Token IDs** — input layer. 2. **Token Embedding** — embedding layer. 2. **Transformer Encoder (6 layers)** — transformer-encoder layer. 2. **Pooler** — linear layer. 2. **Pooled Output** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Distilled BERT: 6-layer transformer encoder trained via knowledge distillation from BERT-base.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
