# Diffusion U-Net (Stable Diffusion)

## Overview

The latent-diffusion noise predictor behind Stable Diffusion: a U-Net operating on 64x64 latents, conditioned on the timestep and on text embeddings via cross-attention at every resolution.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The latent-diffusion noise predictor behind Stable Diffusion: a U-Net operating on 64x64 latents, conditioned on the timestep and on text embeddings via cross-attention at every resolution.

## Key Characteristics

- Cross-attention is the text-conditioning mechanism: image latents attend to CLIP text embeddings inside each block.
- Works in 4x64x64 VAE latent space rather than pixels, which is the "latent" in latent diffusion and why it fits on consumer GPUs.
- This is a compact reference graph of the conditioning topology, not a parameter-faithful replica of the ~860M UNet.

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** CreativeML OpenRAIL-M (SD 1.x weights). The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
