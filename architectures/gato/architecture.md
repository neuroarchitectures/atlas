# Architecture: Gato

## Motivation

DeepMind generalist agent: single transformer processes text, images, button presses, and continuous actions.

## Core Idea

DeepMind generalist agent: single transformer processes text, images, button presses, and continuous actions.

## Architecture

### Overview

![gato architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Tokenized Multi-modal Input | `input` | shape=[2048] |
| 2 | Token Embedding | `embedding` | vocabSize=32000, dim=2048 |
| 3 | Transformer Decoder (24 layers) | `transformer-decoder` | numLayers=24, hiddenSize=2048, numHeads=16 |
| 4 | Output Head | `linear` | outFeatures=32000 |
| 5 | Action / Text Output | `output` | — |

</details>

### Components

1. **Tokenized Multi-modal Input** — input layer. 2. **Token Embedding** — embedding layer. 2. **Transformer Decoder (24 layers)** — transformer-decoder layer. 2. **Output Head** — linear layer. 2. **Action / Text Output** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — DeepMind generalist agent: single transformer processes text, images, button presses, and continuous actions.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
