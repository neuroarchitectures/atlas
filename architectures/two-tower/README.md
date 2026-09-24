# Two-Tower Retrieval

## Overview

The dual-encoder retrieval architecture behind virtually every industrial recommender and semantic search stack: user tower and item tower embed into the same space, scored by dot product.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The dual-encoder retrieval architecture behind virtually every industrial recommender and semantic search stack: user tower and item tower embed into the same space, scored by dot product.

## Key Characteristics

- The towers never interact until the final dot product, which is what makes billion-item retrieval feasible (items pre-embedded into an ANN index).
- Each tower here is embeddings into an MLP; in production the towers grow but the contract (two encoders, one similarity) stays.

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
