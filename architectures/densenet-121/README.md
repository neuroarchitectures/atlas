# DenseNet-121

## Overview

The network where every layer is connected to every other layer in its block: each layer reads the concatenation of all preceding feature maps. Feature reuse instead of re-learning gives strong accuracy at very few parameters.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The network where every layer is connected to every other layer in its block: each layer reads the concatenation of all preceding feature maps.

## Key Characteristics

- The defining op is the concatenate: a layer adds only a small "growth" (32 channels here) but sees everything before it, so gradients and features flow directly to every layer.
- Transition layers between blocks compress channels with a 1x1 conv + pooling so the concatenation does not explode.
- The opposite design philosophy from [resnet-50](../resnet-50/) (additive skips): DenseNet concatenates instead of adds.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Densely-connected convolutional network |
| Parameters | 8M |
| Stem | 7x7/2 conv + 3x3/2 max-pool |
| Dense blocks | 4 blocks, 6/12/24/16 layers |
| Dense layer | BN → ReLU → 1x1 → BN → ReLU → 3x3, output concatenated to input |
| Transitions | BN → 1x1 conv → avg-pool (halve channels) between blocks |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0 / BSD. The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
