# ConvNeXt-Tiny

## Overview

A pure ConvNet rebuilt with every Transformer-era trick (large 7x7 depthwise kernels, LayerNorm not BatchNorm, GeLU, inverted bottleneck, fewer activations) until it matched Swin on ImageNet. The "ConvNets strike back" architecture.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

A pure ConvNet rebuilt with every Transformer-era trick (large 7x7 depthwise kernels, LayerNorm not BatchNorm, GeLU, inverted bottleneck, fewer activations) until it matched Swin on ImageNet.

## Key Characteristics

- The block is essentially a Transformer block with the attention replaced by a large-kernel depthwise conv: depthwise mixes space, the 1x1s mix channels (an inverted bottleneck), with one LayerNorm and one GeLU.
- No attention anywhere, yet it tracks [swin-tiny](../swin-tiny/) closely, the paper's point about how much of ViT's win was design vs attention.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Modernized convolutional network |
| Parameters | 28M |
| Stem | 4x4/4 patchify conv (96) |
| Stages | 4 stages, depths 3/3/9/3 |
| ConvNeXt block | depthwise 7x7 → LayerNorm → 1x1 expand → GeLU → 1x1 project + residual |
| Downsampling | 2x2/2 conv between stages |

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
