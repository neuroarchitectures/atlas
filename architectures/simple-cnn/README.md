# Simple CNN

## Overview

A LeNet-style starter CNN for image classification: conv-pool stacks into a dense head. The "hello world" graph of computer vision.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

A LeNet-style starter CNN for image classification: conv-pool stacks into a dense head.

## Key Characteristics

- The pattern (convolution for local features, pooling for downsampling, dense layers for classification) is unchanged since 1998.
- Good first graph for watching shape propagation catch kernel/stride mistakes.

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
