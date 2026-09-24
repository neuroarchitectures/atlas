# MobileNetV2

## Overview

The on-device vision workhorse. Its inverted residual block expands to a wide intermediate, does a cheap depthwise 3x3 there, then projects back down through a linear bottleneck, packing accuracy into 3.5M parameters and the staple of mobile / CoreML model zoos.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The on-device vision workhorse.

## Key Characteristics

- Inverted residual: unlike ResNet (wide → narrow → wide), MobileNetV2 goes narrow → wide → narrow, and the skip connects the narrow bottlenecks (where the information lives).
- Linear bottleneck: the projection has no ReLU, because ReLU destroys information in low-dimensional space.
- Depthwise-separable convs (the 3x3 acts per-channel) are what make it cheap; full graph with every block expanded.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Efficient convolutional network |
| Parameters | 3.5M |
| Stem | 3x3/2 conv (32 channels) |
| Body | 17 inverted-residual blocks |
| Inverted residual | expand 1x1 → depthwise 3x3 → project 1x1 (linear bottleneck) |
| Head | 1x1 conv (1280) → global avg pool → FC-1000 |
| Input | 3x224x224 |

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
