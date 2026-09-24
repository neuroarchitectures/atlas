# Llama-3-8B

## Overview

Meta's 8B dense decoder, the de facto baseline architecture that most current open LLMs (including many in this zoo) derive from or compare against.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Meta's 8B dense decoder, the de facto baseline architecture that most current open LLMs (including many in this zoo) derive from or compare against.

## Key Characteristics

- The reference open-weight architecture of 2024: 32 layers, GQA 32:8, SwiGLU 14336, RMSNorm pre-norm. Half the models in this zoo are best described as deltas against this graph.
- Big vocabulary jump over Llama-2: 128256 tokens (tiktoken-style BPE), which moves a meaningful fraction of parameters into the embedding and head.
- rope_theta raised to 500000 for the native 8192-token context.
- Graph imported via the NousResearch mirror because the official repo is gated; the config is byte-identical.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 8B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 8 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 14,336 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 128,256 |
| Max context | 8,192 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Llama 3 Community License. The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
