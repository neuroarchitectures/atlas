# Architecture: Pix2Pix

## Motivation

Conditional GAN for image-to-image translation with U-Net generator and PatchGAN discriminator.

## Core Idea

Conditional GAN for image-to-image translation with U-Net generator and PatchGAN discriminator.

## Architecture

### Overview

![pix2pix architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Source Image | `input` | shape=[3, 256, 256] |
| 2 | U-Net Generator | `conv2d` | outFeatures=3 |
| 3 | PatchGAN Discriminator | `conv2d` | outFeatures=1 |
| 4 | Adversarial Output | `output` | — |

</details>

### Components

1. **Source Image** — input layer. 2. **U-Net Generator** — conv2d layer. 2. **PatchGAN Discriminator** — conv2d layer. 2. **Adversarial Output** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Conditional GAN for image-to-image translation with U-Net generator and PatchGAN discriminator.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
