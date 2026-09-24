# Swin Transformer (Tiny)

## Overview

The Vision Transformer that became a general-purpose backbone. Self-attention runs inside fixed local windows (linear cost in image size), and every other block shifts the windows so information crosses window boundaries, while patch merging builds a CNN-like pyramid.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The Vision Transformer that became a general-purpose backbone.

## Key Characteristics

- Windowed attention makes cost linear in pixels (not quadratic like ViT), so Swin scales to detection/segmentation resolutions.
- The shifted-window trick (alternating W-MSA and SW-MSA) is what lets non-adjacent windows communicate without global attention.
- Hierarchical (4 stages, halving resolution and doubling channels) so it drops into FPN-style dense-prediction heads; compare with the flat [vit-b16](../vit-b16/).

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Hierarchical Vision Transformer |
| Parameters | 28M |
| Patch embed | 4x4 conv (96) |
| Stages | 4 stages, depths 2/2/6/2 |
| Block | Window attention, alternating with shifted-window attention |
| Downsampling | Patch merging (2x2 → linear) between stages |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** MIT. The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
