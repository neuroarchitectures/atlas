# Qwen2.5-7B

## Overview

A 7.6B-parameter decoder-only transformer from the Qwen2.5 family, one of the most widely deployed open-weight Chinese and multilingual LLM series. Llama-style pre-norm decoder with grouped-query attention, RoPE, RMSNorm, and SwiGLU.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

A 7.

## Key Characteristics

- Grouped-query attention with 28 query heads sharing 4 KV heads (7:1 ratio), which keeps the KV cache small at long context.
- Unusually for the Llama lineage, Qwen adds a bias term to the Q, K, V projections (attention bias on, everywhere else off).
- Wide FFN: the SwiGLU intermediate size (18944) is about 5.3x the hidden size, wider than the Llama-2 ratio.
- Native 131072-token context window with rope_theta = 1e6.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 7.6B |
| Layers | 28 |
| Hidden size | 3584 |
| Attention | Grouped-query: 28 query heads, 4 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 18,944 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 152,064 |
| Max context | 131,072 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0. The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
