# ResNet Residual Block

## Overview

The residual unit from ResNet: conv-BN-ReLU twice, plus the identity skip connection that made 100+ layer networks trainable. Arguably the most influential 9 nodes in deep learning.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The residual unit from ResNet: conv-BN-ReLU twice, plus the identity skip connection that made 100+ layer networks trainable.

## Key Characteristics

- The skip connection turns layers into residual functions; every modern transformer residual stream descends from this idea.
- Companion to the full [resnet-50](../resnet-50/) entry, which stacks bottleneck versions of this block 16 times.

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
