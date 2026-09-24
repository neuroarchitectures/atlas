# BGE-base-en-v1.5

## Overview

BAAI's BGE retriever, for a long stretch the top open English embedding model on the MTEB leaderboard. A BERT-base encoder fine-tuned with large-scale contrastive pretraining + instruction tuning; the CLS token is the embedding.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

BAAI's BGE retriever, for a long stretch the top open English embedding model on the MTEB leaderboard.

## Key Characteristics

- Architecturally plain BERT-base; all the lift is the C-Pack training recipe (contrastive pretraining on curated pairs, then task fine-tuning).
- Uses the CLS token as the sentence embedding (unlike all-MiniLM's mean pool), and recommends an instruction prefix for queries.
- The "base" tier balances quality and cost; -small and -large siblings trade the two.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Bidirectional encoder, retrieval embedding |
| Parameters | 109M |
| Layers | 12 |
| Hidden size | 768 |
| Attention | Multi-head: 12 heads |
| FFN | Dense, 3072, GeLU |
| Normalization | LayerNorm, post-norm |
| Pooling | CLS token → 768-dim embedding |
| Vocabulary | 30,522 |
| Max context | 512 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** MIT. The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
