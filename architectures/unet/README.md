# U-Net

## Overview

The encoder-decoder with skip connections that owns biomedical and dense-prediction segmentation. Contracting path captures context, expanding path recovers resolution, skips carry the detail across.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The encoder-decoder with skip connections that owns biomedical and dense-prediction segmentation.

## Key Characteristics

- The long skip connections (encoder level i concatenated into decoder level i) are the defining feature; without them the decoder cannot localize.
- Still the backbone shape of choice ten years on, including inside latent diffusion models (see [diffusion-unet](../diffusion-unet/)).
- This is a compact reference graph of the topology, not a parameter-faithful replica of the 31M original.

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

## Related Architectures

See `references/README.md` for related architectures and research context.
