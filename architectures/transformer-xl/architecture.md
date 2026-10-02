# Architecture: Transformer-XL

## Motivation

Transformer with segment-level recurrence and relative positional encoding for long-context language modeling.

## Core Idea

Transformer with segment-level recurrence and relative positional encoding for long-context language modeling.

## Architecture

### Overview

![transformer-xl architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape=[512] |
| 2 | Token Embedding | `embedding` | vocabSize=32000, dim=1024 |
| 3 | Transformer-XL Decoder (18 layers, segment recurrence) | `transformer-decoder` | numLayers=18, hiddenSize=1024, numHeads=16 |
| 4 | LM Head | `linear` | outFeatures=32000 |
| 5 | Output Logits | `output` | — |

</details>

### Components

1. **Token IDs** — input layer. 2. **Token Embedding** — embedding layer. 2. **Transformer-XL Decoder (18 layers, segment recurrence)** — transformer-decoder layer. 2. **LM Head** — linear layer. 2. **Output Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Transformer with segment-level recurrence and relative positional encoding for long-context language modeling.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
