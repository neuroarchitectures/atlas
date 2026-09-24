# Chinese-RoBERTa-wwm-ext

## Overview

The standard Chinese encoder from HFL: BERT-base architecture trained RoBERTa-style with whole-word masking on extended data. Still the default starting point for Chinese classification, NER, and retrieval fine-tuning.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The standard Chinese encoder from HFL: BERT-base architecture trained RoBERTa-style with whole-word masking on extended data.

## Key Characteristics

- Despite the RoBERTa name, this is architecturally BERT-base: it loads as BertModel with absolute learned positions and post-layer-norm. "RoBERTa" refers to the training recipe (no NSP, dynamic masking), not the architecture.
- Whole-word masking (wwm) masks all WordPiece pieces of a Chinese word together, which matters because Chinese has no whitespace word boundaries.
- 21128-token Chinese WordPiece vocabulary, the de facto standard for Chinese BERT-family checkpoints.
- For a decade of Chinese NLU benchmarks (CLUE, etc.), this checkpoint was the baseline everyone compared against.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Bidirectional encoder (BERT family) |
| Parameters | 102M |
| Layers | 12 |
| Hidden size | 768 |
| Attention | Multi-head: 12 heads |
| FFN | Dense, 3,072, GeLU |
| Normalization | LayerNorm, post-norm |
| Positions | Absolute learned, max 512 |
| Vocabulary | 21,128 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0. The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
