# DeepSeek-V2-Lite

## Overview

The small, runnable member of the DeepSeek-V2 family and the most accessible way to study multi-head latent attention. Same MLA + fine-grained-MoE recipe as DeepSeek-V3, shrunk to 16B total / 2.4B active so it fits on one GPU.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The small, runnable member of the DeepSeek-V2 family and the most accessible way to study multi-head latent attention.

## Key Characteristics

- Multi-head latent attention (MLA): keys and values are compressed to a 512-dim latent per token; at this scale queries skip the extra Q-latent that the 236B V2 and 671B V3 use.
- Fine-grained MoE: 64 routed experts (top-6) plus 2 always-on shared experts, slim 1408-dim each; the first layer stays dense.
- The cheapest entry point to the architecture that defined the DeepSeek line, see [deepseek-v3](../deepseek-v3/) for the full-scale version.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer, sparse MoE (causal LM) |
| Parameters | 15.7B total, 2.4B active |
| Layers | 27 |
| Hidden size | 2048 |
| Attention | Multi-head latent (MLA): 16 heads, KV latent 512 (no Q latent at this size) |
| FFN | MoE: 64 routed experts, top-6 + 2 shared, expert dim 1408; first layer dense |
| Normalization | RMSNorm, pre-norm |
| Positions | Decoupled RoPE |
| Vocabulary | 102,400 |
| Max context | 163,840 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Code MIT; weights under the DeepSeek Model License. The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
