# Architecture: AudioMAE

## Motivation

Masked autoencoder for audio: ViT encoder reconstructs masked audio spectrogram patches.

## Core Idea

Masked autoencoder for audio: ViT encoder reconstructs masked audio spectrogram patches.

## Architecture

### Overview

![audiomae architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Audio Spectrogram | `input` | shape=[1, 128, 1024] |
| 2 | Patch Embedding | `conv2d` | outFeatures=768 |
| 3 | ViT Encoder (12 layers) | `transformer-encoder` | numLayers=12, hiddenSize=768, numHeads=12 |
| 4 | ViT Decoder (reconstruction) | `transformer-encoder` | numLayers=2, hiddenSize=384 |
| 5 | Reconstructed Spectrogram | `output` | — |

</details>

### Components

1. **Audio Spectrogram** — input layer. 2. **Patch Embedding** — conv2d layer. 2. **ViT Encoder (12 layers)** — transformer-encoder layer. 2. **ViT Decoder (reconstruction)** — transformer-encoder layer. 2. **Reconstructed Spectrogram** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Masked autoencoder for audio: ViT encoder reconstructs masked audio spectrogram patches.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
