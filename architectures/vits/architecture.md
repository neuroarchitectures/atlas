# Architecture: VITS

## Motivation

TTS pipelines require separate models for duration prediction, acoustic modeling, and vocoding. VITS unifies these into a single end-to-end model using variational autoencoders with adversarial training.

## Core Idea

Use a posterior encoder (normalizing flow) for alignment, a flow-based decoder for waveform generation, and adversarial training for quality. Stochastic duration predictor models variable-length alignment.

## Architecture

### Overview

![vits architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Text | `input` | shape: [100] (phoneme sequence) |
| 2 | Text Embedding | `embedding` | dim: 192 |
| 3 | Text Encoder | `conv` | Transformer or conv encoder |
| 4 | Duration Predictor | `linear` | stochastic duration prediction |
| 5 | Length Regulator | `identity` | expand by predicted durations |
| 6 | Normalizing Flow | `conv` | invertible flow for alignment |
| 7 | HiFi-GAN Decoder | `convTranspose` | waveform generation |
| 8 | Audio | `output` |  |

</details>

### Components

1. **Posterior encoder** — WaveNet-based encoder that extracts latent representation from ground-truth audio for training-time alignment. 2. **Normalizing flow** — Invertible affine coupling layers that transform the prior distribution to match the posterior. 3. **Stochastic duration predictor** — A flow-based model that predicts duration as a random variable, enabling natural prosody variation. 4. **HiFi-GAN decoder** — Transposed convolution decoder for high-fidelity waveform generation. 5. **Adversarial loss** — Multi-period and multi-scale discriminators for audio quality.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use a posterior encoder (normalizing flow) for alignment, a flow-based decoder for waveform generation, and adversarial training for quality. Stochastic duration predictor models variable-length align
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Glow-TTS, HiFi-GAN. Successor: VITS2, NaturalSpeech.

## References

- Kim et al. 2021
