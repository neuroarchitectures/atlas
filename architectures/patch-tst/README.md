# PatchTST

## Overview

The Transformer that made long-horizon time-series forecasting work: each variable is sliced into patches (like ViT, but in time) and encoded channel-independently by a shared Transformer.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The Transformer that made long-horizon time-series forecasting work: each variable is sliced into patches (like ViT, but in time) and encoded channel-independently by a shared Transformer.

## Key Characteristics

- Patching cuts sequence length quadratically for attention and gives each token local semantic content.
- Channel independence (one shared encoder applied per variable) beat channel-mixing on the standard long-horizon benchmarks.

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
