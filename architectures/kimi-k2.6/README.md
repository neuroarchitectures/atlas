# Kimi K2.6

## Overview

Moonshot AI's trillion-parameter-class MoE, the largest architecture in this zoo. A DeepSeek-V3-style MLA + fine-grained-MoE design pushed to 384 experts and 256K context, with a vision encoder bolted on in K2.6.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Moonshot AI's trillion-parameter-class MoE, the largest architecture in this zoo.

## Key Characteristics

- DeepSeek-V3 lineage scaled out: same 61-layer, 7168-hidden MLA backbone, but 384 routed experts (vs 256), half the attention heads (64), and only the first layer dense.
- HF safetensors metadata puts the total at 1.06T parameters with roughly 32B active per token.
- 262144-token context with rope_theta 50000 on the 64-dim decoupled RoPE part.
- The config carries a vision encoder (patch 14) alongside the text stack: K2.6 is natively multimodal. This entry shows the text decoder.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer, sparse MoE (causal LM) |
| Parameters | ~1.06T total, ~32B active |
| Layers | 61 |
| Hidden size | 7168 |
| Attention | Multi-head latent: 64 heads; KV latent 512, Q latent 1536; per head 128 NoPE + 64 RoPE, V 128 |
| FFN | MoE: 384 routed experts, top-8 + 1 shared, expert dim 2,048; first 1 layer dense (18,432) |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 64) |
| Vocabulary | 163,840 |
| Max context | 262,144 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Modified MIT (K2 family). The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
