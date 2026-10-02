# Architecture: FastSpeech 2

## Motivation

Non-autoregressive TTS with variance adaptor for pitch, energy, and duration prediction.

## Core Idea

Non-autoregressive TTS with variance adaptor for pitch, energy, and duration prediction.

## Architecture

### Overview

![fastspeech-2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Phoneme IDs | `input` | shape=[256] |
| 2 | Phoneme Embedding | `embedding` | vocabSize=100, dim=256 |
| 3 | Transformer Encoder (4 layers) | `transformer-encoder` | numLayers=4, hiddenSize=256, numHeads=2 |
| 4 | Variance Adaptor (duration + pitch + energy) | `linear` | outFeatures=256 |
| 5 | Transformer Decoder (4 layers) | `transformer-decoder` | numLayers=4, hiddenSize=256, numHeads=2 |
| 6 | Mel-spectrogram Head | `linear` | outFeatures=80 |
| 7 | Mel-spectrogram | `output` | — |

</details>

### Components

1. **Phoneme IDs** — input layer. 2. **Phoneme Embedding** — embedding layer. 2. **Transformer Encoder (4 layers)** — transformer-encoder layer. 2. **Variance Adaptor (duration + pitch + energy)** — linear layer. 2. **Transformer Decoder (4 layers)** — transformer-decoder layer. 2. **Mel-spectrogram Head** — linear layer. 2. **Mel-spectrogram** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Non-autoregressive TTS with variance adaptor for pitch, energy, and duration prediction.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
