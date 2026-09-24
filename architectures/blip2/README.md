# BLIP-2

## Overview

Bridges a frozen image encoder and a frozen LLM with a lightweight Querying Transformer. A fixed set of 32 learned query tokens cross-attend the image features, distilling them into something the language model can read, so almost no parameters are trained.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Bridges a frozen image encoder and a frozen LLM with a lightweight Querying Transformer.

## Key Characteristics

- The Q-Former is the whole idea: learned queries that self-attend each other and cross-attend the frozen image features, outputting a fixed 32-token visual summary.
- Both the vision encoder and the LLM stay frozen; only the small Q-Former (and a linear projection) are trained, which is why BLIP-2 is so cheap to build.
- A different bridge from [llava-1.5-7b](../llava-1.5-7b/)'s simple MLP projector, a learned cross-attention compressor vs a direct projection.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Multimodal (image-to-text) |
| Vision encoder | Frozen ViT |
| Bridge | Q-Former: 32 learned queries, self + cross-attention to image |
| LLM | Frozen (OPT / Flan-T5), fed the projected queries |
| Trained | Only the Q-Former (everything else frozen) |

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
