# BERT4Rec

## Overview

The recsys answer to BERT: a bidirectional Transformer over the item sequence trained with masked-item (Cloze) prediction. Each item sees both its past and future neighbours, unlike causal SASRec.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The recsys answer to BERT: a bidirectional Transformer over the item sequence trained with masked-item (Cloze) prediction.

## Key Characteristics

- Bidirectional self-attention plus a masked-item training objective (randomly mask items, predict them from both sides).
- The both-directions context can capture patterns a left-to-right model misses, at the cost of the Cloze-style training/serving mismatch.
- Directly comparable to [sasrec](../sasrec/): same inputs, bidirectional-masked vs causal.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Sequential recommendation |
| Embedding | Item + learned positional |
| Backbone | Bidirectional Transformer encoder |
| Objective | Masked-item (Cloze) prediction |
| Key idea | Both-directions context, BERT-style |

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
