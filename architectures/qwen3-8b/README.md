# Qwen3-8B

## Overview

The dense 8B from the Qwen3 generation, the default open model of 2025 by download count. Architecturally a cleanup: QKV bias out, QK-Norm in, more depth per parameter.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The dense 8B from the Qwen3 generation, the default open model of 2025 by download count.

## Key Characteristics

- QK-Norm replaces the Qwen2.5 QKV bias: per-head RMSNorm on queries and keys before RoPE (attention_bias is false in the config), trading the bias trick for norm-based attention stability.
- Deeper and thinner than Qwen2.5-7B: 36 layers at 4096 hidden versus 28 at 3584, with a narrower 12288 FFN.
- GQA 32:8 with explicit head_dim 128; native 40960-token context at rope_theta 1e6.
- The dense workhorse of the Qwen3 release; the same recipe scales to the 235B-A22B MoE flagship. Hybrid thinking/non-thinking modes are a training-time property, not an architecture change.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 8.2B |
| Layers | 36 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 8 KV heads, QK-Norm |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 12,288 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 151,936 |
| Max context | 40,960 |

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
