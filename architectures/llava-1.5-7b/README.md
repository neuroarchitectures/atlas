# LLaVA-1.5-7B

## Overview

The canonical recipe for making an LLM see: a frozen CLIP vision encoder, a tiny MLP projector that maps image-patch features into the LLM's token embedding space, and a Llama decoder that attends over image and text tokens jointly. Simple, and it defined the open multimodal-LLM playbook.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

The canonical recipe for making an LLM see: a frozen CLIP vision encoder, a tiny MLP projector that maps image-patch features into the LLM's token embedding space, and a Llama decoder that attends over image and text tokens jointly.

## Key Characteristics

- The whole trick is the projector: just a 2-layer MLP turning 576 image-patch vectors into 576 "visual tokens" the LLM can read; only it (and the LLM) are trained, the vision encoder stays frozen.
- Visual tokens are concatenated in front of the text tokens, so the unmodified Llama decoder treats the image as a prefix.
- This bridge-into-the-LLM pattern is what most open MLLMs (Qwen-VL, InternVL, ...) still follow.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Type | Multimodal LLM (vision + language) |
| Vision encoder | CLIP ViT-L/14-336 (frozen): 24 blocks, 1024 hidden |
| Projector | 2-layer MLP, 1024 → 4096 (the bridge) |
| LLM | Vicuna/Llama-7B: 32 decoder blocks, 4096 hidden |
| Fusion | Image tokens prepended to text tokens |
| Parameters | ~7B (mostly the LLM) |

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Full structural model graph (nodes, layers, parameters, tensor shapes). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, official repos, related work). |

**License:** Llama 2 Community License (LLM); Apache 2.0 (code). The graph and diagrams here describe the architecture; any referenced weights remain under the upstream license.

## Related Architectures

See `references/README.md` for related architectures and research context.
