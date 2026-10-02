# Architecture: BigVGAN

## Motivation

Existing neural vocoders produce artifacts at unseen pitch frequencies due to aliasing in nonlinear activations.

## Core Idea

Introduce anti-aliased activation functions (low-pass filtered activations) combined with snake activation to reduce spectral artifacts, enabling universal vocoding across unseen speakers and languages.

## Architecture

### Overview

![bigvgan architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Mel Spectrogram | `input` | shape: [80] |
| 2 | Upsample | `conv1d` | stride: 8 |
| 3 | Anti-Aliased ResBlock | `residual-block` |  |
| 4 | Upsample | `conv1d` | stride: 8 |
| 5 | Anti-Aliased ResBlock | `residual-block` |  |
| 6 | Output Conv | `conv1d` |  |
| 7 | Waveform | `output` |  |

</details>

### Components

1. **Anti-aliased activation** — Low-pass filter applied before activation to prevent aliasing. 2. **Snake activation** — a + sin²(bx) provides periodic inductive bias for audio. 3. **Multi-period discriminator** — Adversarial training at multiple periods.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Introduce anti-aliased activation functions (low-pass filtered activations) combined with snake activation to reduce spe
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: HiFi-GAN, UnivNet. Successor: Vocos.

## References

Lee et al. 2023
