# Architecture: Whisper Large

## Motivation

Speech recognition systems are typically trained on narrow domains. Multilingual robustness requires massive-scale weak supervision.

## Core Idea

Train an encoder-decoder transformer on 680K hours of multilingual audio collected from the web, using multitask supervision (transcription, translation, language ID, timestamping).

## Architecture

### Overview

![whisper-large architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Mel Spectrogram | `input` | shape: [80, 3000] |
| 2 | Conv1d | `conv1d` |  |
| 3 | Conv1d | `conv1d` |  |
| 4 | Transformer Encoder | `transformer` | 32 layers |
| 5 | Transformer Decoder | `transformer` | 32 layers |
| 6 | Transcription Tokens | `output` |  |

</details>

### Components

1. **Mel spectrogram input** — 80-channel log-mel filterbank at 100Hz. 2. **Convolutional stem** — Two Conv1d layers for local feature extraction. 3. **Transformer encoder** — 32-layer encoder (1280 dim, 20 heads). 4. **Transformer decoder** — 32-layer autoregressive decoder with special tokens for task specification.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Train an encoder-decoder transformer on 680K hours of multilingual audio collected from the web, using multitask supervi
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: wav2vec 2.0, HuBERT. Successor: Whisper v3, Distil-Whisper.

## References

Radford et al. 2022
