# all-MiniLM-L6-v2

## Overview

The most-downloaded sentence-embedding model in the world: a 6-layer MiniLM distilled from BERT, mean-pooled into a 384-dim vector. The default workhorse for semantic search, RAG retrieval, and clustering when you want fast and small.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The most-downloaded sentence-embedding model in the world: a 6-layer MiniLM distilled from BERT, mean-pooled into a 384-dim vector.

## Key Characteristics

- Tiny BERT encoder (6 layers, 384 hidden) distilled via MiniLM's deep-self-attention distillation, then fine-tuned on 1B+ sentence pairs with a contrastive objective.
- The embedding is a mean pool over the token outputs (not the CLS token), then L2-normalized, so cosine similarity ranks relevance.
- At ~80MB and 384 dims it is the cheap retriever most RAG stacks start with.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Bidirectional encoder, sentence embedding |
| Parameters | 22.7M |
| Layers | 6 |
| Hidden size | 384 |
| Attention | Multi-head: 12 heads |
| FFN | Dense, 1536, GeLU |
| Normalization | LayerNorm, post-norm |
| Pooling | Mean over tokens → 384-dim sentence vector |
| Vocabulary | 30,522 |
| Max context | 256 (trained), 512 (max) |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0. The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
