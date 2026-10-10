# Architecture: Qwen-VL

## Motivation

VLMs need to understand documents, OCR, and grounding (bounding boxes), not just image-text matching. Qwen-VL adds position-aware adapters and multi-level encoding for fine-grained visual understanding.

## Core Idea

ViT encodes the image. A position-aware vision adapter (cross-attention with 2D position embeddings) compresses visual tokens while preserving spatial info. Qwen LLM processes visual + text tokens. Can output text and bounding box coordinates.

## Architecture

### Overview

![qwen-vl architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3x448x448 |
| 2 | ViT Encoder | `conv2d` | ViT-L/14 |
| 3 | Position-aware Adapter | `linear` | cross-attn + 2D pos |
| 4 | Text Embedding | `linear` | Qwen embeddings |
| 5 | Concat | `identity` | visual + text |
| 6 | Qwen LLM | `linear` | 7B params |
| 7 | Text + Bounding Boxes | `output` | text + grounding |

</details>

### Components

1. **ViT-L/14** — vision encoder from CLIP. 2. **Position-aware adapter** — cross-attention with 2D position embeddings. 3. **Token compression** — reduces 256 visual tokens to 64. 4. **Multi-level encoding** — encodes at multiple resolutions. 5. **Bounding box output** — can ground objects with coordinates. 6. **OCR capability** — trained on document/text data.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — ViT encodes the image. A position-aware vision adapter (cross-attention with 2D position embeddings) compresses visual t...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Qwen-LLM, LLaVA. Successor: Qwen-VL2, Qwen2-VL.

## References

- Bai et al. 2023
