# NCF (concat-MLP)

## Overview

Neural Collaborative Filtering in its simplest form: user and item embeddings concatenated into an MLP that learns the interaction function instead of assuming a dot product.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Neural Collaborative Filtering in its simplest form: user and item embeddings concatenated into an MLP that learns the interaction function instead of assuming a dot product.

## Key Characteristics

- The pure-MLP variant of the NCF paper; the fused GMF+MLP variant is the separate [neumf](../neumf/) entry.
- The paper's claim that an MLP beats the inner product sparked a years-long replication debate (Rendle et al. 2020); keep both graphs around to test it yourself.

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
