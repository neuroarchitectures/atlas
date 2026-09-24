# SASRec

## Overview

Self-Attentive Sequential Recommendation: a causal Transformer over the user's item history that predicts the next item, GPT for recommendation. One of the most-cited and most-reimplemented sequential-recsys baselines.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Self-Attentive Sequential Recommendation: a causal Transformer over the user's item history that predicts the next item, GPT for recommendation.

## Key Characteristics

- Causal self-attention means position t attends only to items up to t, exactly like a language-model decoder.
- Adaptively weights which past items matter for the next click, instead of the fixed recency bias of an RNN.
- Pairs with [bert4rec](../bert4rec/): same item-sequence setup, causal (SASRec) vs bidirectional masked (BERT4Rec).

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Sequential recommendation |
| Embedding | Item + learned positional |
| Backbone | Causal (left-to-right) self-attention blocks |
| Objective | Next-item prediction |
| Key idea | Self-attention over history, GPT-style |

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
