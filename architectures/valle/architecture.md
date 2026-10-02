# Architecture: VALL-E

## Motivation

Neural codec language model for TTS: zero-shot voice cloning from 3-second enrollment audio.

## Core Idea

Neural codec language model for TTS: zero-shot voice cloning from 3-second enrollment audio.

## Architecture

### Overview

![valle architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Phoneme + Acoustic Tokens | `input` | shape=[1024] |
| 2 | Token Embedding | `embedding` | vocabSize=1024, dim=1024 |
| 3 | Transformer Decoder (12 layers) | `transformer-decoder` | numLayers=12, hiddenSize=1024, numHeads=16 |
| 4 | LM Head | `linear` | outFeatures=1024 |
| 5 | Acoustic Tokens | `output` | — |

</details>

### Components

1. **Phoneme + Acoustic Tokens** — input layer. 2. **Token Embedding** — embedding layer. 2. **Transformer Decoder (12 layers)** — transformer-decoder layer. 2. **LM Head** — linear layer. 2. **Acoustic Tokens** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Neural codec language model for TTS: zero-shot voice cloning from 3-second enrollment audio.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
