# DeepSeek-V2

## Overview

The 236B MoE where multi-head latent attention and fine-grained mixture-of-experts debuted, the architecture the whole DeepSeek line is built on. Cheap KV cache from MLA, cheap compute from 160 slim experts.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The 236B MoE where multi-head latent attention and fine-grained mixture-of-experts debuted, the architecture the whole DeepSeek line is built on.

## Key Characteristics

- The model that introduced multi-head latent attention and fine-grained MoE, the recipe DeepSeek-V3 later scaled. 236B total, only 21B active per token.
- MLA: KV compressed to a 512-dim latent, Q to a 1536-dim latent; each of 128 heads splits into 128-dim NoPE content + 64-dim decoupled RoPE.
- 160 routed experts (top-6) + 2 shared, slim 1536-dim each; first layer dense.
- See [deepseek-v2-lite](../deepseek-v2-lite/) for the runnable 16B version and [deepseek-v3](../deepseek-v3/) for the scaled-up 671B.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer, sparse MoE (causal LM) |
| Parameters | 236B total, 21B active |
| Layers | 60 |
| Hidden size | 5120 |
| Attention | Multi-head latent: 128 heads; KV latent 512, Q latent 1536; per head 128 NoPE + 64 RoPE, V 128 |
| FFN | MoE: 160 routed experts, top-6 + 1 shared, expert dim 1,536; first 1 layer dense (12,288) |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 64) |
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

**License:** Code MIT; weights under the DeepSeek Model License. The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
