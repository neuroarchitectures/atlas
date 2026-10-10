# Architecture: Chameleon

## Motivation

Meta early-fusion multimodal model: all modalities tokenized and processed as one sequence.

## Core Idea

Meta early-fusion multimodal model: all modalities tokenized and processed as one sequence.

## Architecture

### Overview

![chameleon architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Tokenized Multi-modal Input | `input` | shape=[4096] |
| 2 | Token Embedding | `embedding` | vocabSize=65536, dim=4096 |
| 3 | Transformer Decoder (32 layers) | `transformer-decoder` | numLayers=32, hiddenSize=4096, numHeads=32 |
| 4 | Output Head | `linear` | outFeatures=65536 |
| 5 | Multi-modal Output | `output` | — |

</details>

### Components

1. **Tokenized Multi-modal Input** — input layer. 2. **Token Embedding** — embedding layer. 2. **Transformer Decoder (32 layers)** — transformer-decoder layer. 2. **Output Head** — linear layer. 2. **Multi-modal Output** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Meta early-fusion multimodal model: all modalities tokenized and processed as one sequence.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
