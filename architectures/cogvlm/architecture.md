# Architecture: CogVLM

## Motivation

Existing VLMs either shallowly fuse vision (project image features into LLM embedding space) or deeply fuse but hurt language performance. CogVLM introduces a visual expert (separate FFN + attention params for image tokens) in every layer for deep fusion without degrading text.

## Core Idea

EVA-CLIP encodes the image. A trainable projection maps visual features to LLM dimension. The LLM (CogLM) has a visual expert module in every attention layer — separate parameters for image tokens that are activated alongside the standard pathway.

## Architecture

### Overview

![cogvlm architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3x490x490 |
| 2 | EVA-CLIP Encoder | `conv2d` | ViT-L/14 |
| 3 | Visual Projection | `linear` | 1024 -> 4096 |
| 4 | Text Embedding | `linear` | LLM embeddings |
| 5 | Concat | `identity` | image + text tokens |
| 6 | LLM with Visual Expert | `linear` | CogLM, visual expert per layer |
| 7 | Text Output | `output` | autoregressive |

</details>

### Components

1. **EVA-CLIP vision encoder** — frozen ViT-L/14. 2. **Visual projection** — trainable MLP maps to LLM space. 3. **Visual expert** — separate FFN + attention parameters for image tokens in every layer. 4. **Dual pathway** — text tokens use standard params, image tokens use expert params. 5. **Deep fusion** — image features influence every layer, not just input. 6. **No degradation** — text-only performance preserved.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — EVA-CLIP encodes the image. A trainable projection maps visual features to LLM dimension. The LLM (CogLM) has a visual e...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: LLaVA (shallow fusion), BLIP-2. Successor: CogVLM2, InternVL.

## References

- Wang et al. 2023
