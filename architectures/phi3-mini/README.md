# Phi-3 Mini Block

## Overview

The Phi-3 Mini (3.8B) decoder block: a compact Llama-style block at 3072 hidden, the architecture behind the "small model trained on textbook-quality data" line of work.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The Phi-3 Mini (3.

## Key Characteristics

- Full multi-head attention (32 heads at 3072, head dim 96); no GQA at this scale in the 4k variant.
- SwiGLU FFN at 8192 intermediate, RMSNorm pre-norm, RoPE.
- Architecturally conventional on purpose: the Phi thesis is that data quality, not architecture novelty, drives small-model performance.

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

## Related Architectures

See `references/README.md` for related architectures and research context.
