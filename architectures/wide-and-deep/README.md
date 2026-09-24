# Wide & Deep

## Overview

Google's Play-store ranking model that joint-trains a wide linear path (memorization of cross features) with a deep embedding MLP (generalization), summed at the logit.

> Architecture diagram and model graph migrated from [awesome-llm-model-zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo). See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Google's Play-store ranking model that joint-trains a wide linear path (memorization of cross features) with a deep embedding MLP (generalization), summed at the logit.

## Key Characteristics

- The wide path memorizes specific feature crosses; the deep path generalizes to unseen ones; the sum gets both behaviors in one model.
- The conceptual ancestor of DeepFM, DCN, and most hybrid CTR architectures since.

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
