# ViT-B/16

## Overview

The Vision Transformer that ended CNN hegemony in image classification: 16x16 patch embedding, learned position embeddings, and a stack of standard pre-norm Transformer encoder blocks.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The Vision Transformer that ended CNN hegemony in image classification: 16x16 patch embedding, learned position embeddings, and a stack of standard pre-norm Transformer encoder blocks.

## Key Characteristics

- The patch-embed stem is just a strided conv: 3x224x224 becomes 196 patch tokens of 768 dims (plus the class token).
- Identical block to BERT but pre-norm; the inductive-bias-free design needs large-scale pretraining to win.
- The full 12-block stack lives in model.json; the block view shows one encoder block expanded.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Vision Transformer (image classification) |
| Parameters | 86M |
| Layers | 12 encoder blocks |
| Hidden size | 768 |
| Attention | Multi-head: 12 heads |
| FFN | Dense MLP, 3072, GeLU |
| Normalization | LayerNorm, pre-norm |
| Positions | Learned, 196 patches + class token |
| Patch size | 16x16 over 224x224 input |

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
