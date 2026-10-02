# Architecture: FastSpeech

## Motivation

Autoregressive TTS (Tacotron 2) is slow at inference due to sequential mel generation. FastSpeech uses a non-autoregressive feed-forward transformer to generate all mel frames in parallel, achieving ~38x speedup.

## Core Idea

Phoneme embeddings pass through a feed-forward transformer (FFT) encoder. A duration predictor estimates how many mel frames each phoneme spans. A length regulator expands phoneme features to mel length. An FFT decoder generates mel-spectrograms in one forward pass.

## Architecture

### Overview

![fastspeech architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Phoneme Sequence | `input` | phoneme ids |
| 2 | Phoneme Embedding | `linear` | 384 dim |
| 3 | FFT Encoder | `linear` | 4 FFT blocks |
| 4 | Duration Predictor | `identity` | predicts frame count |
| 5 | Length Regulator | `identity` | expands to mel length |
| 6 | FFT Decoder | `linear` | 4 FFT blocks |
| 7 | Mel Spectrogram | `linear` | 80 mel bins |
| 8 | Mel Output | `output` | mel frames |

</details>

### Components

1. **Feed-forward transformer (FFT)** — self-attention + FFN blocks replacing recurrent decoder. 2. **Duration predictor** — CNN predicting phoneme duration from encoder output. 3. **Length regulator** — upsamples phoneme-level features to frame-level by repeating. 4. **Non-autoregressive generation** — all mel frames produced simultaneously.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Phoneme embeddings pass through a feed-forward transformer (FFT) encoder. A duration predictor estimates how many mel fr...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Tacotron 2 (autoregressive), Transformer TTS. Successor: FastSpeech 2 (better variance modeling), VITS (end-to-end).

## References

- Ren et al. 2019
