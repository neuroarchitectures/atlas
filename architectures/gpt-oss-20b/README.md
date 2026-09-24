# gpt-oss-20b

## Overview

The 20B mixture-of-experts OpenAI released under Apache 2.0 in 2025. Small sliding windows, tiny heads, aggressive MoE sparsity: an inference-economics architecture through and through.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The 20B mixture-of-experts OpenAI released under Apache 2.

## Key Characteristics

- OpenAI's first open-weight release since GPT-2: a 24-layer MoE with 32 experts, top-4 routed, no shared expert, 3.6B active of 21B total.
- Alternating attention: odd layers use a 128-token sliding window, even layers full attention (verified from config layer_types, 12 of each), plus learned attention-sink logits per head.
- Attention bias on, 64 small heads of dim 64 over a 2880 hidden size, and a 201088-token o200k_harmony vocabulary.
- Ships MXFP4-quantized so the 20B fits in 16GB; trained with the harmony response format for tool use and CoT.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer, sparse MoE (causal LM) |
| Parameters | 21B total, 3.6B active |
| Layers | 24 |
| Hidden size | 2880 |
| Attention | Grouped-query: 64 query heads, 8 KV heads |
| Head dim | 64 |
| FFN | MoE: 32 routed experts, top-4, expert dim 2,880 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE + YaRN; layers alternate sliding-window (128) and full attention 1:1 |
| Vocabulary | 201,088 |
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
