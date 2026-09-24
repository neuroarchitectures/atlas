# VGG-16

## Overview

The architecture that proved depth plus uniform 3x3 convolutions beats clever filter engineering. Sixteen weight layers in five conv stages with the classic 138M-parameter dense head.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The architecture that proved depth plus uniform 3x3 convolutions beats clever filter engineering.

## Key Characteristics

- Design rule: only 3x3 convs and 2x2 max-pools, doubling channels after each pool (64 to 512). Two stacked 3x3 convs see a 5x5 field with fewer parameters and more nonlinearity.
- Roughly 124M of the 138M parameters sit in the three dense layers, the inefficiency that GAP-based heads (ResNet onward) eliminated.
- Still everywhere as a perceptual-loss and style-transfer feature extractor.

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
