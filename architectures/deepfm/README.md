# DeepFM

## Overview

The CTR model that fused a factorization machine with a deep network under one shared embedding table. FM captures low-order feature interactions, the MLP captures high-order, and unlike Wide & Deep neither path needs hand-engineered crosses.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The CTR model that fused a factorization machine with a deep network under one shared embedding table.

## Key Characteristics

- Both branches read the SAME embedding table, so the FM and deep components are trained jointly without separate feature engineering.
- The FM branch models explicit second-order (pairwise) interactions; the deep MLP models higher-order ones implicitly.
- The ancestor of a whole family (xDeepFM, AutoInt, DCN); pairs with [wide-and-deep](../wide-and-deep/) as the "learned vs hand-crafted crosses" comparison.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | CTR / click prediction |
| FM branch | Shared embeddings → pairwise (2nd-order) interactions |
| Deep branch | Same embeddings flattened → MLP (high-order) |
| Fusion | Sum of the two logits → sigmoid |
| Key idea | No manual feature crosses (vs Wide & Deep) |

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
