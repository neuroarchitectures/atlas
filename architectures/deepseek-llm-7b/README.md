# DeepSeek-LLM-7B

## Overview

The original dense 7B model from DeepSeek, trained on 2T bilingual tokens. A clean Llama-style decoder (RoPE, RMSNorm, SwiGLU) with a large bilingual vocabulary, and the architectural starting point of the DeepSeek lineage.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The original dense 7B model from DeepSeek, trained on 2T bilingual tokens.

## Key Characteristics

- The dense ancestor of the DeepSeek series, before the MoE turn (V2/V3). Architecture is deliberately Llama-compatible: model_type in config.json is literally "llama".
- Plain multi-head attention at 7B (32 Q = 32 KV heads); only the 67B sibling adopted grouped-query attention.
- 30 layers instead of the usual 32, slightly shallower and wider than Llama-2-7B.
- Large 102400-token byte-level BPE vocabulary tuned for bilingual Chinese and English text.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 6.9B |
| Layers | 30 |
| Hidden size | 4096 |
| Attention | Multi-head: 32 heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 11,008 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 102,400 |
| Max context | 4,096 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Code MIT; weights under the DeepSeek Model License (commercial use permitted). The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
