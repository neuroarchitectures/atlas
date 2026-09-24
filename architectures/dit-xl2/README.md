# DiT-XL/2

## Overview

The model that replaced the diffusion U-Net with a plain Transformer over latent patches, conditioned by adaptive LayerNorm. DiT scaled cleanly and became the backbone of Stable Diffusion 3, PixArt, and Sora-class video models.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The model that replaced the diffusion U-Net with a plain Transformer over latent patches, conditioned by adaptive LayerNorm.

## Key Characteristics

- adaLN-zero: each block's LayerNorm scale/shift (and residual gates) are produced by a linear from the timestep + class embedding, so conditioning enters through normalization rather than cross-attention.
- Operates in VAE latent space (4x32x32) on 2x2 patches, exactly the ViT recipe applied to a denoiser.
- Replacing the U-Net (see [diffusion-unet](../diffusion-unet/)) with a Transformer is why diffusion now scales like LLMs do.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Diffusion model, Transformer backbone |
| Parameters | 675M (XL/2) |
| Layers | 28 DiT blocks |
| Hidden size | 1152 |
| Attention | Multi-head: 16 heads |
| Conditioning | adaLN-zero from timestep + class embedding |
| Patches | 2x2 over a 32x32x4 latent |
| FFN | Dense MLP, 4608, GeLU |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** CC-BY-NC (weights). The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
