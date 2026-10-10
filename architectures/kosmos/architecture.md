# Architecture: Kosmos-2

## Motivation

Multimodal LLM with grounded text-image alignment: vision encoder + language model for referring expressions.

## Core Idea

Multimodal LLM with grounded text-image alignment: vision encoder + language model for referring expressions.

## Architecture

### Overview

![kosmos architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image + Text | `input` | shape=[3, 224, 224] |
| 2 | ViT Vision Encoder | `transformer-encoder` | numLayers=14, hiddenSize=1024, numHeads=16 |
| 3 | Language Model Decoder | `transformer-decoder` | numLayers=24, hiddenSize=2048, numHeads=16 |
| 4 | Text Output | `output` | — |

</details>

### Components

1. **Image + Text** — input layer. 2. **ViT Vision Encoder** — transformer-encoder layer. 2. **Language Model Decoder** — transformer-decoder layer. 2. **Text Output** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Multimodal LLM with grounded text-image alignment: vision encoder + language model for referring expressions.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
