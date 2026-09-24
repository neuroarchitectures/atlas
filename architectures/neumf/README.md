# NeuMF (GMF + MLP)

## Overview

The full NCF model: a Generalized Matrix Factorization path (element-wise product of embeddings) fused with an MLP path, each with its own embedding tables, joined before the final score.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The full NCF model: a Generalized Matrix Factorization path (element-wise product of embeddings) fused with an MLP path, each with its own embedding tables, joined before the final score.

## Key Characteristics

- Two separate embedding sets, one per path, concatenated at the last layer: the structural detail most reimplementations get wrong.
- GMF preserves the classic MF inductive bias while the MLP path adds flexibility; the fusion is the paper's actual headline model.

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
