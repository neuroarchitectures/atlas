# Baichuan2-7B

## Overview

Baichuan Inc.'s second-generation 7B base model, a Llama-shaped pre-norm decoder distinguished by a large Chinese-optimized vocabulary and a normalized output head (NormHead).

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Baichuan Inc.

## Key Characteristics

- Llama-2-7B shape (32 layers, 4096 hidden, 11008 FFN) with a much larger 125696-token vocabulary optimized for Chinese.
- NormHead: the output embedding (LM head) rows are L2-normalized before the logit matmul, which the tech report credits with stabilizing training.
- The 7B model uses RoPE; do not confuse it with the 13B sibling, which uses ALiBi instead.
- Trained on 2.6T tokens; one of the first Chinese LLMs to ship intermediate training checkpoints for research.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 7.5B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Multi-head: 32 heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 11,008 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 125,696 |
| Max context | 4,096 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Structural model graph (nodes, layers, parameters, tensor shapes); 32 identical decoder layers compressed via a NAXS block template + `repeat`. |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Code Apache 2.0; weights under the Baichuan2 Community License (free commercial use after application). The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
