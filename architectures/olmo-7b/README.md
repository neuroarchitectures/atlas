# OLMo-7B

## Overview

Ai2's fully-open 7B: not just open weights but open training data (Dolma), code, and logs. A Llama-shaped decoder with one signature twist, non-parametric LayerNorm, included as the reference for reproducible LLM research.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Ai2's fully-open 7B: not just open weights but open training data (Dolma), code, and logs.

## Key Characteristics

- Non-parametric LayerNorm: the norm has no learnable gain or bias at all, just the normalization, which the OLMo report found improved stability.
- Otherwise a clean Llama-style decoder: RoPE, SwiGLU, no biases, untied embeddings.
- The point is end-to-end openness (Dolma corpus + training code + checkpoints), making it the model to use when you need to know exactly what went in.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 6.9B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Multi-head: 32 heads |
| FFN | SwiGLU, intermediate size 11008 |
| Normalization | Non-parametric LayerNorm (no weight / bias) |
| Positions | RoPE |
| Vocabulary | 50,304 |
| Max context | 4,096 |

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
