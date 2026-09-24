# Falcon-7B

## Overview

TII's 2023 open LLM, briefly the top of the Open LLM Leaderboard. Two architecture bets define it: multi-query attention (all 71 query heads share a single key/value head, shrinking the KV cache ~70x) and a parallel residual block where one LayerNorm feeds attention and the MLP at once, both summed back.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

TII's 2023 open LLM, briefly the top of the Open LLM Leaderboard.

## Key Characteristics

- Multi-query attention: 71 query heads but only 1 KV head, the extreme end of the MHA -> GQA -> MQA spectrum.
- Parallel attention + MLP (GPT-NeoX style): both consume the same normed input and add into the residual, saving a norm per layer.
- GELU MLP (not gated SwiGLU) and no biases anywhere; RoPE for positions.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 7.2B |
| Layers | 32 |
| Hidden size | 4,544 |
| Attention | Multi-query: 71 query heads, 1 KV head, head dim 64 |
| FFN | GELU MLP, intermediate size 18,176 (4x) |
| Residual | Parallel: one LayerNorm feeds attention and MLP, both summed |
| Normalization | LayerNorm, pre-norm |
| Positions | RoPE |
| Vocabulary | 65,024 |
| Max context | 2,048 |

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
