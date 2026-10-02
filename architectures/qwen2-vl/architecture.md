# Architecture: Qwen2-VL

## Motivation

Existing VLMs resize images to fixed resolution, losing information. Position encodings struggle with varying sequence lengths from different image sizes.

## Core Idea

Use a dynamic resolution vision encoder that processes images at native resolution, combined with Multimodal Rotary Position Embedding (M-RoPE) for handling temporal, height, and width dimensions.

## Architecture

### Overview

![qwen2-vl architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 1024, 1024] |
| 2 | Text | `input` |  |
| 3 | ViT | `vit` | dynamic resolution |
| 4 | Visual Projection | `linear` |  |
| 5 | Qwen2 LLM | `transformer` | 28 layers |
| 6 | Text Output | `output` |  |

</details>

### Components

1. **Dynamic ViT** — Processes images at arbitrary resolution by adaptive patching. 2. **Visual projection** — Maps visual features to LLM embedding space. 3. **Qwen2 LLM** — Transformer decoder with M-RoPE. 4. **M-RoPE** — Decomposes position into temporal, height, width components.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use a dynamic resolution vision encoder that processes images at native resolution, combined with Multimodal Rotary Posi
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Qwen-VL. Successor: Qwen2.5-VL, InternVL2.

## References

Wang et al. 2024
