# ERNIE 3.0 Base (Chinese)

## Overview

Baidu's workhorse Chinese encoder, the base-size distillation of the ERNIE 3.0 family. BERT-base shape with a larger vocabulary, 2048-token positions, and knowledge-enhanced pretraining.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Baidu's workhorse Chinese encoder, the base-size distillation of the ERNIE 3.

## Key Characteristics

- BERT-base shape with two ERNIE twists: a 40000-token vocabulary (almost 2x BERT's Chinese vocab) and a 2048 max position, four times BERT's 512.
- Adds a task-type embedding alongside token and position embeddings, a remnant of ERNIE 3.0's multi-task universal-representation pretraining.
- Distilled from the 10B ERNIE 3.0 Titan teacher; the base model is what ships for practical NLU.
- Official weights are Paddle-native; the linked HF checkpoint is the standard PyTorch conversion by nghuyong.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Bidirectional encoder (BERT family) |
| Parameters | 118M |
| Layers | 12 |
| Hidden size | 768 |
| Attention | Multi-head: 12 heads |
| FFN | Dense, 3,072, GeLU |
| Normalization | LayerNorm, post-norm |
| Positions | Absolute learned, max 2,048 |
| Vocabulary | 40,000 |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Apache 2.0 (PaddleNLP). The graph and diagrams here describe the architecture; the model weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
