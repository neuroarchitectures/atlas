# Architecture: Latent Diffusion

## Motivation

Pixel-space diffusion (DDPM) is computationally expensive for high-resolution images because the U-Net operates on full-resolution pixels. Latent Diffusion compresses images into a lower-dimensional latent space using an autoencoder, then performs diffusion there.

## Core Idea

Two-stage approach: (1) Train a VAE autoencoder that compresses images to a low-dimensional latent space (e.g., HxWx4 vs HxWx3), (2) Train a diffusion model (U-Net or DiT) in this latent space. This reduces computation by ~8x while maintaining quality, enabling high-resolution generation.

## Architecture

### Overview

![latent-diffusion architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Latent Noise z_T | `input` | shape: [1, 4, 32, 32] |
| 2 | U-Net Denoiser | `custom` | type: unet, inChannels: 4, modelChannels: 320, attentionResolutions: [4, 2, 1] |
| 3 | Denoised Latent z_0 | `custom` | type: denoised_latent |
| 4 | VAE Decoder | `custom` | type: vae_decoder, inChannels: 4, outChannels: 3 |
| 5 | Generated Image | `output` |  |
| 6 | Training Image | `input` | shape: [1, 3, 256, 256] |
| 7 | VAE Encoder | `custom` | type: vae_encoder, inChannels: 3, outChannels: 4 |
| 8 | Latent z | `custom` | type: encoded_latent |
| 9 | Add Noise | `custom` | type: forward_diffusion, steps: 1000 |

</details>

Two-stage approach: (1) Train a VAE autoencoder that compresses images to a low-dimensional latent space (e.g., HxWx4 vs HxWx3), (2) Train a diffusion model (U-Net or DiT) in this latent space. This reduces computation by ~8x while maintaining quality, enabling high-resolution generation.

### Components

2. **U-Net Denoiser** (`custom`, scope: `diffusion`) — Params: type: unet, inChannels: 4, modelChannels: 320, attentionResolutions: [4, 2, 1]
3. **Denoised Latent z_0** (`custom`, scope: `diffusion`) — Params: type: denoised_latent
4. **VAE Decoder** (`custom`, scope: `vae`) — Params: type: vae_decoder, inChannels: 4, outChannels: 3
7. **VAE Encoder** (`custom`, scope: `vae`) — Params: type: vae_encoder, inChannels: 3, outChannels: 4
8. **Latent z** (`custom`, scope: `vae`) — Params: type: encoded_latent
9. **Add Noise** (`custom`, scope: `diffusion`) — Params: type: forward_diffusion, steps: 1000

### Data Flow

The architecture processes input through a sequence of 9 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Latent Diffusion.

## Evolution

Latent Diffusion (Stable Diffusion) builds on DDPM/DDIM by moving diffusion to latent space. It is the basis for Stable Diffusion 1.x/2.x/3.x, FLUX, and most modern text-to-image systems. Successors include DiT (transformer denoiser), SDXL, and rectified flow models.

## Source

- **Paper:** arXiv:2112.10752
- **Year:** 2022
- **Authors:** Rombach et al.
