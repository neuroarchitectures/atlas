# Architecture: WavLM

## Motivation

Pre-trained speech representation model with denoising masking for robust speech features.

## Core Idea

Pre-trained speech representation model with denoising masking for robust speech features.

## Architecture

### Overview

![wavlm architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Audio Waveform | `input` | shape=[1, 16000] |
| 2 | 1D CNN Feature Extractor | `conv1d` | outFeatures=512 |
| 3 | Transformer Encoder (12 layers) | `transformer-encoder` | numLayers=12, hiddenSize=768, numHeads=12 |
| 4 | Speech Features | `output` | — |

</details>

### Components

1. **Audio Waveform** — input layer. 2. **1D CNN Feature Extractor** — conv1d layer. 2. **Transformer Encoder (12 layers)** — transformer-encoder layer. 2. **Speech Features** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Pre-trained speech representation model with denoising masking for robust speech features.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
