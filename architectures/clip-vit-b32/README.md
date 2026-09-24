# CLIP ViT-B/32

## Overview

The contrastive image-text model that underpins modern multimodality: a full 12-block ViT image tower and a full 12-block Transformer text tower projected into one 512-dim space, trained so matching pairs score high by dot product. Stable Diffusion's text encoder is the text half of this design.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The contrastive image-text model that underpins modern multimodality: a full 12-block ViT image tower and a full 12-block Transformer text tower projected into one 512-dim space, trained so matching pairs score high by dot product.

## Key Characteristics

- Two towers, one space: each tower ends in a linear projection to 512 dims; the similarity logit is a scaled dot product.
- The graph pools each tower with an average-pool node as a stand-in for CLIP's token selection (class token for the image tower, EOT token for text).
- Tower dims verified from the official config.json; the text tower is causal, a quirk inherited from GPT-style pretraining.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Contrastive image-text dual encoder |
| Parameters | 151M (88M vision + 63M text) |
| Vision tower | ViT-B/32: 12 blocks, 768 hidden, 12 heads |
| Text tower | 12 blocks, 512 hidden, 8 heads, causal |
| Projection | Both towers project to a shared 512-dim space |
| Similarity | Scaled dot product, learned temperature |
| Vocabulary | 49,408 BPE (text), 77-token max |
| Input | 224x224 image, 32x32 patches |

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
