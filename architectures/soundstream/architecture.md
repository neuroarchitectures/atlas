# Architecture: SoundStream

## Motivation

Audio compression codecs (Opus, EVS) use hand-engineered signal processing. SoundStream replaces this with a learned encoder-decoder and residual vector quantization, achieving competitive quality at lower bitrates.

## Core Idea

Strided convolutions downsample the waveform to a compact latent representation. Residual vector quantization (RVQ) discretizes the latent space with multiple cascaded codebooks. Transposed convolutions upsample back to waveform. Adversarial + reconstruction losses train the system end-to-end.

## Architecture

### Overview

![soundstream architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Waveform | `input` | 24kHz mono |
| 2 | Encoder | `conv1d` | strided downsampling |
| 3 | RVQ Quantizer | `identity` | cascaded VQ codebooks |
| 4 | Decoder | `conv1d` | transposed upsampling |
| 5 | Reconstructed Waveform | `output` | 24kHz mono |

</details>

### Components

1. **Strided encoder** — convolutional downsampling by factor ~320. 2. **Residual vector quantization (RVQ)** — multiple VQ layers, each quantizing the residual from the previous. 3. **Decoder** — transposed convolutions reconstruct waveform. 4. **Discriminators** — multi-scale and multi-period adversarial training. 5. **Joint training** — encoder, quantizer, and decoder trained together.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Strided convolutions downsample the waveform to a compact latent representation. Residual vector quantization (RVQ) disc...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: VQ-VAE (discrete latent), EnCodec. Successor: AudioLM (uses SoundStream tokens), MusicLM.

## References

- Zeghidour et al. 2021
