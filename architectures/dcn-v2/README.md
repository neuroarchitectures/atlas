# DCN-v2

## Overview

The production successor to DCN. Each cross layer applies a full weight matrix to the feature vector before the element-wise cross with the input (x0 . (W x_l) + x_l), so the crosses are far more expressive than DCN's rank-1 version, while a low-rank factorization keeps the cost down. Deployed across Google ad and feed ranking.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The production successor to DCN.

## Key Characteristics

- The defining change from [dcn](../dcn/): the cross uses a learned matrix W (here a full linear), where the original used a rank-1 weight vector.
- Reference topology: features projected to a base vector x0, two matrix-cross layers, a parallel deep MLP, then concatenation.
- In practice the cross matrix is low-rank (a bottleneck) to control parameters at production feature counts.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | CTR / click prediction |
| Cross network | Matrix-weighted cross per layer, plus residual |
| Deep network | Parallel MLP |
| Fusion | Concatenate cross + deep then logit |
| Key idea | Full (low-rank) cross matrix replaces DCN's rank-1 cross |

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
