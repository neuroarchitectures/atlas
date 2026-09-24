# Behavior Sequence Transformer

## Overview

Alibaba's CTR model that replaced sum-pooled user history with a Transformer encoder over the click sequence, with the candidate item appended as the final sequence element.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Alibaba's CTR model that replaced sum-pooled user history with a Transformer encoder over the click sequence, with the candidate item appended as the final sequence element.

## Key Characteristics

- The candidate-in-sequence trick lets self-attention compute target-aware interest weights directly.
- Deployed in Taobao ranking; one of the first production proofs that Transformers transfer to recsys sequences.

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
