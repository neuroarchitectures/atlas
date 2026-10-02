# Architecture: Stable Diffusion

## Motivation

Diffusion in pixel space is computationally expensive for high-resolution images. Stable Diffusion performs diffusion in a compressed latent space (4x64x64 instead of 3x512x512), making it feasible on consumer GPUs.

## Core Idea

A VAE autoencoder compresses images to a 4x64x64 latent space. A UNet denoiser performs iterative denoising in this latent space, conditioned on CLIP text embeddings. The VAE decoder upsamples the denoised latent back to pixel space.

## Architecture

### Overview

![stable-diffusion architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Latent Noise | `input` | 4x64x64 |
| 2 | Text Prompt | `input` | text -> CLIP embeddings |
| 3 | CLIP Text Encoder | `linear` | 77 tokens x 768 dim |
| 4 | UNet Denoiser | `conv2d` | denoise in latent space |
| 5 | VAE Decoder | `conv2d` | latent -> 3x512x512 |
| 6 | Generated Image | `output` | 512x512 |

</details>

### Components

1. **VAE autoencoder** — compresses 3x512x512 to 4x64x64 latent. 2. **UNet denoiser** — iterative denoising in latent space (20-50 steps). 3. **CLIP text encoder** — text conditioning via cross-attention. 4. **Latent diffusion** — diffusion operates on 4-channel latents, not pixels. 5. **Classifier-free guidance** — conditional + unconditional predictions combined. 6. **DPM-Solver** — faster sampling schedules.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A VAE autoencoder compresses images to a 4x64x64 latent space. A UNet denoiser performs iterative denoising in this late...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Latent Diffusion (Rombach 2022), DALL-E 2. Successor: SDXL, SD3.

## References

- Rombach et al. 2022
