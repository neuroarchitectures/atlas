# MiniCPM-2B

## Overview

OpenBMB's flagship small LLM, a 2.4B deep-and-thin decoder that punches above its weight via muP-style scaling (scale_emb, scaled residuals, tied embeddings) found by extensive scaling-law sweeps.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

OpenBMB's flagship small LLM, a 2.

## Key Characteristics

- Deep-and-thin: 40 layers at only 2304 hidden, the opposite trade-off from most 2B-class models.
- muP-style stability tricks baked into the architecture: embedding output scaled by scale_emb = 12, residual branches scaled by scale_depth / sqrt(num_layers), and tied input/output embeddings.
- Plain multi-head attention (36 heads, 64-dim each); the KV-cache saving tricks were left out at this scale.
- A 122753-token vocabulary, very large for a 2B model, reflecting its Chinese-first design.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 2.7B total (2.4B non-embedding) |
| Layers | 40 |
| Hidden size | 2304 |
| Attention | Multi-head: 36 heads |
| Head dim | 64 |
| FFN | SwiGLU, intermediate size 5,760 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 64) |
| Vocabulary | 122,753 |
| Max context | 4,096 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0 (code); weights free for commercial use after registration. The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
