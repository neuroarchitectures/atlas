# EfficientNet-B0

## Overview

The baseline of the EfficientNet family: MBConv blocks (mobile inverted bottleneck with a squeeze-and-excite gate) discovered by neural architecture search, then compound-scaled into B1-B7. The accuracy-per-FLOP reference for years.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The baseline of the EfficientNet family: MBConv blocks (mobile inverted bottleneck with a squeeze-and-excite gate) discovered by neural architecture search, then compound-scaled into B1-B7.

## Key Characteristics

- MBConv = MobileNetV2's inverted residual plus a squeeze-and-excite channel-attention block inside each one.
- B0 was found by NAS; B1-B7 just scale depth, width, and resolution together by a single compound coefficient.
- Compare with [mobilenet-v2](../mobilenet-v2/) (same inverted-residual core, no SE) and [resnet-50](../resnet-50/) (the non-inverted predecessor).

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Efficient convolutional network |
| Parameters | 5.3M |
| Stem | 3x3/2 conv (32) |
| Body | 16 MBConv blocks (7 stages) |
| MBConv | expand 1x1 → depthwise → squeeze-excite → project 1x1 |
| Head | 1x1 conv (1280) → GAP → FC-1000 |
| Found by | Compound-scaling NAS |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0. The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
