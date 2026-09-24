# BERT-Base

## Overview

The bidirectional encoder that started the transfer-learning era in NLP. Twelve post-norm encoder blocks, 768 hidden, 12 heads: the shape every "base-size" encoder since has copied.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The bidirectional encoder that started the transfer-learning era in NLP.

## Key Characteristics

- The model that made pretrain-then-finetune the default workflow in NLP (Devlin et al. 2018, arXiv 1810.04805).
- Post-norm placement: LayerNorm comes after each residual add, the original Transformer ordering that pre-norm models later abandoned for training stability.
- Learned absolute position embeddings hard-cap the context at 512 tokens.
- 30522-token WordPiece vocabulary; masked-language-modeling plus next-sentence-prediction pretraining.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Bidirectional encoder (BERT family) |
| Parameters | 110M |
| Layers | 12 |
| Hidden size | 768 |
| Attention | Multi-head: 12 heads |
| FFN | Dense, 3,072, GeLU |
| Normalization | LayerNorm, post-norm |
| Positions | Absolute learned, max 512 |
| Vocabulary | 30,522 |

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
