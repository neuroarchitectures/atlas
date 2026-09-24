# Mistral-7B

## Overview

The 7B model from Mistral AI that beat Llama-2-13B at release, built on grouped-query attention plus sliding-window attention. The direct dense ancestor of Mixtral.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The 7B model from Mistral AI that beat Llama-2-13B at release, built on grouped-query attention plus sliding-window attention.

## Key Characteristics

- Sliding-window attention: each layer attends to the previous 4096 tokens only; stacking 32 layers grows the effective receptive field to the full 32768-token context. The graph shows standard GQA; the window is an attention-mask detail.
- Grouped-query attention 32:8 plus the sliding window made it the fastest-inference 7B of its generation.
- Kept the compact Llama-2 32000-token vocabulary while adopting the larger 14336 FFN.
- The dense base that Mixtral 8x7B sparsified: the MoE model reuses this exact block with the FFN swapped for 8 experts.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 7.2B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 8 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 14,336 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 32,000 |
| Max context | 32,768 |

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
