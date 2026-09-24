# HuBERT base

## Overview

Architecturally a twin of Wav2Vec2 (conv feature extractor + Transformer), but pretrained differently: instead of contrastive masking it predicts the cluster ID (from offline k-means on features) of masked frames, a BERT-style masked-prediction objective for speech.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Architecturally a twin of Wav2Vec2 (conv feature extractor + Transformer), but pretrained differently: instead of contrastive masking it predicts the cluster ID (from offline k-means on features) of masked frames, a BERT-style masked-prediction objective for speech.

## Key Characteristics

- Same backbone as [wav2vec2-base](../wav2vec2-base/); the contribution is the training target, a discrete unit from iterative k-means clustering rather than a contrastive loss.
- That "predict the masked unit" recipe (closer to masked-LM) ended up underpinning a lot of modern speech and audio-LLM systems.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Self-supervised speech encoder |
| Parameters | 95M |
| Feature extractor | 7 Conv1D layers (raw waveform) |
| Encoder | 12 Transformer blocks, 768 hidden |
| Pretraining | Predict offline k-means cluster IDs of masked frames |

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
