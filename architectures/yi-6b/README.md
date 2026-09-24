# Yi-6B

## Overview

The 6B base model from 01.AI (founded by Kai-Fu Lee), trained on 3.1T bilingual tokens. A Llama-style decoder notable for using grouped-query attention at small scale and for its 200K long-context variant.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The 6B base model from 01.

## Key Characteristics

- Fully Llama-compatible architecture (model_type "llama"), so the whole Llama tooling ecosystem works out of the box.
- Grouped-query attention even at 6B scale: 32 query heads over 4 KV heads, unusual for a model this small at the time.
- High rope_theta (5e6) chosen with long-context extension in mind; the Yi-6B-200K variant stretches the same architecture to 200k tokens.
- Compact 64000-token bilingual vocabulary.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Decoder-only transformer (causal LM) |
| Parameters | 6B |
| Layers | 32 |
| Hidden size | 4096 |
| Attention | Grouped-query: 32 query heads, 4 KV heads |
| Head dim | 128 |
| FFN | SwiGLU, intermediate size 11,008 |
| Normalization | RMSNorm, pre-norm |
| Positions | RoPE (rotary dim 128) |
| Vocabulary | 64,000 |
| Max context | 4,096 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0 (relicensed from the Yi License in 2024). The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
