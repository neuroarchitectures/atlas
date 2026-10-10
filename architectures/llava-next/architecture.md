# Architecture: LLaVA-NeXT

## Motivation

LLaVA-1.5 uses fixed-resolution images and a simple linear projection, limiting performance on text-rich images (OCR, charts).

## Core Idea

Use anyres visual encoding that divides images into a grid of sub-images processed independently, combined with an MLP projector for richer visual features.

## Architecture

### Overview

![llava-next architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 672, 672] |
| 2 | Text | `input` |  |
| 3 | CLIP ViT (AnyRes) | `vit` | 24 layers |
| 4 | MLP Projection | `mlp` |  |
| 5 | LLM | `transformer` | 32 layers |
| 6 | Text Output | `output` |  |

</details>

### Components

1. **AnyRes ViT** — Divides image into 2x2 or 1x4 grid, each tile processed by CLIP ViT. 2. **MLP projection** — Two-layer MLP projects visual features to LLM space. 3. **LLM** — Autoregressive transformer (7B-34B). 4. **Token aggregation** — Visual tokens from all tiles concatenated.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use anyres visual encoding that divides images into a grid of sub-images processed independently, combined with an MLP p
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: LLaVA-1.5. Successor: LLaVA-OneVision.

## References

Liu et al. 2024
