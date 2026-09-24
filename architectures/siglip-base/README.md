# SigLIP base

## Overview

CLIP with one change that matters: a pairwise sigmoid loss instead of the softmax-contrastive one. Dropping the batch-global normalization lets it train well at any batch size, and its vision tower is the encoder many 2024+ multimodal LLMs (including PaliGemma) build on.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

CLIP with one change that matters: a pairwise sigmoid loss instead of the softmax-contrastive one.

## Key Characteristics

- The architecture is a CLIP-style dual encoder; the contribution is the sigmoid loss, which treats every image-text pair as an independent binary problem.
- No softmax over the batch means no need for the huge batches CLIP relied on, and the text tower drops the causal mask CLIP inherited from GPT.
- Compare with [clip-vit-b32](../clip-vit-b32/): same two-tower shape, different objective.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Contrastive image-text dual encoder |
| Parameters | 203M |
| Vision tower | ViT-B/16: 12 blocks, 768 hidden, 12 heads |
| Text tower | 12 blocks, 768 hidden (no causal mask) |
| Loss | Pairwise sigmoid (not softmax contrastive) |
| Pooling | Attention/mean pool per tower |

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
