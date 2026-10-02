# Architecture: SDXL

## Motivation

Stable Diffusion 2.x is limited to 512x512 resolution and uses a single text encoder, limiting prompt understanding and image quality.

## Core Idea

Scale up the UNet, use dual text encoders (CLIP + OpenCLIP), process at 1024x1024 natively, and add an optional refinement model for high-frequency detail.

## Architecture

### Overview

![sdxl architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Latent | `input` | shape: [4, 128, 128] |
| 2 | Text | `input` |  |
| 3 | CLIP Text Encoder | `clip-text` | 1280-dim |
| 4 | OpenCLIP Text Encoder | `clip-text` | 1024-dim |
| 5 | Concat Text Features | `concat` |  |
| 6 | UNet | `unet` |  |
| 7 | VAE Decoder | `vae` |  |
| 8 | Image | `output` |  |

</details>

### Components

1. **Dual text encoders** — CLIP ViT-L (1280-dim) + OpenCLIP ViT-bigG (1024-dim), concatenated. 2. **Large UNet** — More attention blocks, bigger channels. 3. **1024x1024 native** — Processes at higher resolution without upsampling. 4. **Refinement model** — Optional second-stage diffusion for detail enhancement.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Scale up the UNet, use dual text encoders (CLIP + OpenCLIP), process at 1024x1024 natively, and add an optional refineme
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Stable Diffusion 2.1. Successor: SD3, SDXL Turbo.

## References

Podell et al. 2024
