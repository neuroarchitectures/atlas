# GRU4Rec

## Overview

The paper that brought recurrent networks to session-based recommendation: a GRU consumes the sequence of clicks in an anonymous session and scores the next item. The RNN baseline every later sequential-recsys model compares against.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The paper that brought recurrent networks to session-based recommendation: a GRU consumes the sequence of clicks in an anonymous session and scores the next item.

## Key Characteristics

- Built for session data (no long-term user profile): the GRU's hidden state is the running session intent.
- Introduced session-parallel mini-batching and a ranking loss (BPR / TOP1) tailored to recommendation.
- The recurrent counterpart to the attention-based [sasrec](../sasrec/) and [bert4rec](../bert4rec/).

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Session-based recommendation |
| Embedding | Item embedding |
| Backbone | GRU over the click sequence |
| Head | Linear to item catalogue → softmax |
| Key idea | RNN for anonymous session sequences |

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
