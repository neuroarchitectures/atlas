# SLi-Rec

## Overview

Short- and Long-term interest Recommender: a Time-aware LSTM models the evolving short-term interest while an attentive ASVD component captures stable long-term preference, fused adaptively.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Short- and Long-term interest Recommender: a Time-aware LSTM models the evolving short-term interest while an attentive ASVD component captures stable long-term preference, fused adaptively.

## Key Characteristics

- Time-aware LSTM gates take the irregular gaps between user actions as input, not just the order.
- The adaptive fusion learns per-user, per-moment weighting between the long- and short-term signals.

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
