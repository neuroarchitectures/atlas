# Architecture: DeBERTa

## Motivation

Disentangled attention encoder with relative position encoding and enhanced mask decoder.

## Core Idea

Disentangled attention encoder with relative position encoding and enhanced mask decoder.

## Architecture

### Overview

![deberta architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Token IDs | `input` | shape=[512] |
| 2 | Token Embedding | `embedding` | vocabSize=50265, dim=768 |
| 3 | Disentangled Attention Encoder | `transformer-encoder` | numLayers=12, hiddenSize=768, numHeads=12 |
| 4 | Enhanced Mask Decoder | `transformer-encoder` | numLayers=1, hiddenSize=768 |
| 5 | Output Logits | `output` | — |

</details>

### Components

1. **Token IDs** — input layer. 2. **Token Embedding** — embedding layer. 2. **Disentangled Attention Encoder** — transformer-encoder layer. 2. **Enhanced Mask Decoder** — transformer-encoder layer. 2. **Output Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Disentangled attention encoder with relative position encoding and enhanced mask decoder.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
