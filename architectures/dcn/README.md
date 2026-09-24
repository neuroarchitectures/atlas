# DCN (Deep & Cross)

## Overview

Google's Deep & Cross Network: a cross network that applies an explicit feature-crossing formula at every layer (so L layers give degree-L crosses), run alongside a deep MLP and concatenated. Generalizes DeepFM's fixed second-order FM to arbitrary order.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Google's Deep & Cross Network: a cross network that applies an explicit feature-crossing formula at every layer (so L layers give degree-L crosses), run alongside a deep MLP and concatenated.

## Key Characteristics

- Each cross layer computes x0 · xl^T · w + xl (a residual feature cross), so stacking k layers yields explicit crosses up to degree k+1 with very few parameters.
- The cross and deep networks run in parallel and are concatenated before the final logit.
- DCN-v2 later swapped the rank-1 cross for a low-rank matrix; this is the original formulation.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | CTR / click prediction |
| Cross network | Stacked layers, each an explicit feature cross + residual |
| Deep network | Parallel MLP |
| Fusion | Concatenate cross + deep → logit → sigmoid |
| Key idea | Bounded-degree feature crosses learned explicitly |

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
