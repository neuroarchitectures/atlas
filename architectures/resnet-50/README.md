# ResNet-50

## Overview

The most-cited convolutional network ever and the ILSVRC 2015 winner. This is the full graph: conv stem, all 16 bottleneck residual blocks (3+4+6+3) with every conv, batch norm, and skip connection, global average pooling, and the 1000-way head.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The most-cited convolutional network ever and the ILSVRC 2015 winner.

## Key Characteristics

- Bottleneck design: 1x1 reduce, 3x3 conv, 1x1 expand in every block, which is how 50 layers stay at 25.6M parameters.
- Stage transitions double channels and halve resolution with strided 1x1 projections on the skip path.
- A decade later it is still the default vision backbone for detection, segmentation, and as a sanity-check baseline.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Convolutional network (image classification) |
| Parameters | 25.6M |
| Stem | 7x7/2 conv (64) + 3x3/2 max-pool |
| Stages | 4 stages of bottleneck blocks: 3, 4, 6, 3 |
| Bottleneck | 1x1 reduce, 3x3, 1x1 expand (4x) |
| Channels | 256 / 512 / 1024 / 2048 per stage |
| Shortcuts | Identity; 1x1 projection at stage boundaries |
| Head | Global average pool + FC-1000 |
| Input | 3x224x224 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0 (HF weights). The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
