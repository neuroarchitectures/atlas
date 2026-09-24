# MPT-7B

## Overview

MosaicML's 2023 open, commercially-usable LLM. Its defining choice is ALiBi: instead of learned or rotary position embeddings, it biases attention scores by a linear penalty on key-query distance, which lets the model extrapolate to context lengths well beyond training. No biases, tied embeddings, plain GELU MLP.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

MosaicML's 2023 open, commercially-usable LLM.

## Key Characteristics

- ALiBi (Attention with Linear Biases): no positional embedding tensor at all; a fixed per-head linear distance penalty is added to attention scores, enabling length extrapolation.
- No biases anywhere and tied input/output embeddings, keeping the parameter count tight.
- Standard multi-head attention with a 128 head dim; a clean baseline for the ALiBi positional approach.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 6.6B |
| Layers | 32 |
| Hidden size | 4,096 |
| Attention | Multi-head: 32 heads, head dim 128 |
| FFN | GELU MLP, intermediate size 16,384 (4x) |
| Normalization | LayerNorm, pre-norm |
| Positions | ALiBi (linear attention bias, no positional embeddings) |
| Vocabulary | 50,432 |
| Max context | 2,048 trained; extrapolates via ALiBi |

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
