# LightGCN

## Overview

Graph convolution for collaborative filtering stripped to its minimum: no feature transforms, no nonlinearities, just neighborhood propagation over the user-item graph and a layer-wise average.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Graph convolution for collaborative filtering stripped to its minimum: no feature transforms, no nonlinearities, just neighborhood propagation over the user-item graph and a layer-wise average.

## Key Characteristics

- An ablation result turned architecture: removing the transforms and activations from NGCF improved accuracy.
- Final embeddings average over propagation layers (layer combination), capturing multi-hop signals at different ranges.

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
