# Mixtral MoE Block

## Overview

The Mixtral 8x7B decoder block: the Mistral-7B block with its dense FFN swapped for a sparse mixture of 8 expert FFNs, top-2 routed per token.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The Mixtral 8x7B decoder block: the Mistral-7B block with its dense FFN swapped for a sparse mixture of 8 expert FFNs, top-2 routed per token.

## Key Characteristics

- Sparse MoE layer: 8 experts, top-2 routing, so roughly 13B of 47B total parameters are active per token.
- Everything around the MoE layer is the Mistral-7B recipe: GQA 32:8 with RoPE, RMSNorm pre-norm, dual residual streams.
- The canonical open-weight example of decoupling parameter count from inference FLOPs.

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
