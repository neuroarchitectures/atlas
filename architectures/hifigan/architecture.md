# Architecture: HiFi-GAN

## Motivation

Autoregressive models (WaveNet) are slow at inference. Parallel models (WaveGlow) are large. HiFi-GAN achieves both fast inference and high audio quality using a non-autoregressive GAN.

## Core Idea

Generator uses transposed convolutions for upsampling plus multi-receptive field fusion blocks. Two discriminators: Multi-Period Discriminator (operates on reshaped 2D patterns) and Multi-Scale Discriminator (operates at different resolutions).

## Architecture

### Overview

![hifi-gan architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Mel Spectrogram | `input` | shape: [80, 256] |
| 2 | Upsample 1 (8x) | `convTranspose` | 8x upsampling |
| 3 | MRF Block 1 | `conv` | Multi-Receptive Field fusion |
| 4 | Upsample 2 (8x) | `convTranspose` | 8x upsampling |
| 5 | MRF Block 2 | `conv` | Multi-Receptive Field fusion |
| 6 | Output Conv | `conv` | kernelSize: 7, to 1-channel audio |
| 7 | Audio Waveform | `output` |  |

</details>

### Components

1. **Generator** — Transposed convolutions for upsampling (total 256x from mel to audio). Multi-Receptive Field (MRF) blocks: parallel dilated convolutions with different kernel/dilation, summed. 2. **Multi-Period Discriminator (MPD)** — Reshapes 1D audio to 2D with different periods (2, 3, 5, 7, 11), then applies 2D convolutions to capture periodic patterns. 3. **Multi-Scale Discriminator (MSD)** — Operates on the raw waveform at different downsampling scales. 4. **Losses** — Adversarial loss + feature matching loss + mel-spectrogram L1 loss.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Generator uses transposed convolutions for upsampling plus multi-receptive field fusion blocks. Two discriminators: Multi-Period Discriminator (operates on reshaped 2D patterns) and Multi-Scale Discri
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: WaveNet, WaveGlow. Successor: BigVGAN, VITS.

## References

- Kong et al. 2021
