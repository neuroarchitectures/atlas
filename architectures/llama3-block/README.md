# Llama-3 Decoder Block

## Overview

A single Llama-3 decoder block at 8B dimensions, expanded to individual operations: RMSNorm, grouped-query attention with RoPE, residual add, RMSNorm, SwiGLU FFN, residual add. The companion to the full [llama3-8b](../llama3-8b/) entry.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

A single Llama-3 decoder block at 8B dimensions, expanded to individual operations: RMSNorm, grouped-query attention with RoPE, residual add, RMSNorm, SwiGLU FFN, residual add.

## Key Characteristics

- Shows the block internals that the full-model entry collapses: both residual streams, the pre-norm placement, and RoPE feeding the attention node.
- GQA 32 query heads over 8 KV heads; SwiGLU intermediate size 14336.
- Useful as a starting graph when you want to modify the block itself (try MoE, different norms, attention variants) and re-validate.

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
