# InternLM2-7B

## Overview

Shanghai AI Laboratory's second-generation 7B base model. A pre-norm decoder with 32:8 grouped-query attention and a native 32k context window, plus strong tool-use and long-context chat variants.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Shanghai AI Laboratory's second-generation 7B base model.

## Key Characteristics

- Llama-3-like shape a year early: 32 layers, GQA 32:8, and the same 14336 FFN intermediate size.
- Native 32768-token context trained with rope_theta = 1e6, extrapolating to 200k in the chat variants.
- Consolidated W_qkv weight layout: q, k, v are stored interleaved per KV group, which matters when converting checkpoints.
- 92544-token vocabulary covering Chinese, English, and code.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 7.7B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 8 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 14,336 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 92,544 |
| Max context | 32,768 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Code Apache 2.0; weights free for commercial use after application. The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
