# DIN (Deep Interest Network)

## Overview

Alibaba's production CTR model that replaced fixed sum-pooling of a user's behaviour history with an attention "activation unit": each past behaviour is weighted by how relevant it is to the candidate item, so the user's interest vector is target-aware.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Alibaba's production CTR model that replaced fixed sum-pooling of a user's behaviour history with an attention "activation unit": each past behaviour is weighted by how relevant it is to the candidate item, so the user's interest vector is target-aware.

## Key Characteristics

- The activation unit is a local attention between the candidate item and each historical behaviour; high-relevance behaviours dominate the pooled interest.
- This is the recsys insight that "a user's interest is multi-modal, attend to the part relevant to what you are scoring".
- Followed by DIEN (adds a GRU-based interest-evolution layer); compare with [bst](../bst/), which instead runs a full Transformer over the behaviour sequence.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | CTR with user-behavior sequence |
| Activation unit | Attention of behaviours w.r.t. the candidate item |
| Interest | Relevance-weighted sum of behaviours (not mean pool) |
| Head | Concat interest + candidate → MLP → sigmoid |
| Key idea | Target-aware interest representation |

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
