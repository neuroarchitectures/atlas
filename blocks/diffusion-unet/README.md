# Diffusion U-Net (Latent Diffusion Denoiser)

## Design Philosophy

A U-Net operating on **VAE latents** (not pixels), conditioned on the timestep and on text embeddings via **cross-attention** at every resolution. The philosophy: working in a compressed latent space (e.g., 4×64×64) instead of pixels is why latent diffusion fits on consumer GPUs, and cross-attention is the text-conditioning mechanism. Replaced by DiT (Transformer) in Stable Diffusion 3.

## Functionality

- **conv_in**: 3×3 conv from latent channels to base width (320).
- **Down blocks**: ResNet blocks + cross-attention (image latents attend to CLIP text embeddings) at each resolution; downsample by 2×.
- **Mid block**: ResNet + cross-attention + ResNet.
- **Up blocks**: ResNet + cross-attention + skip-concat from the matching down block; upsample by 2×.
- **conv_out**: 3×3 conv back to latent channels (predicts noise).
- **Conditioning**: Timestep → SiLU → linear (added to ResNet blocks); text → cross-attention.
- **GroupNorm** (not BatchNorm) throughout.

## Used By

| Model | Role |
|-------|------|
| Stable Diffusion 1.x / 2.x | The noise predictor (~860M UNet) |
- The latent-diffusion topology; DiT replaces it in SD3.

## Features

- **Latent space**: 4×64×64 instead of 3×512×512 — the "latent" in latent diffusion.
- **Cross-attention conditioning**: Image latents attend to text embeddings at every block.
- **Timestep conditioning**: Injected via SiLU+linear into every ResNet block.

## Evolution

- **Predecessor**: U-Net (segmentation); DDPM (pixel-space diffusion).
- **Successor**: DiT (Transformer over latent patches, adaLN conditioning) — SD3, Sora.
