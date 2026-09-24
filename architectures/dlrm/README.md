# DLRM

## Overview

Meta's production recommendation architecture: a bottom MLP for dense features, embedding tables for sparse features, explicit pairwise dot-product interactions, and a top MLP over the result.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Meta's production recommendation architecture: a bottom MLP for dense features, embedding tables for sparse features, explicit pairwise dot-product interactions, and a top MLP over the result.

## Key Characteristics

- The explicit pairwise interaction layer (dot products between all embedding pairs) is the signature; it is factorization machines absorbed into a deep net.
- At production scale the embedding tables dominate parameters so heavily that DLRM training is a memory-bandwidth problem, which shaped a generation of recsys hardware.

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
