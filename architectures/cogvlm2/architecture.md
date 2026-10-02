# Architecture: CogVLM2

## Motivation

Shallow visual-language fusion (projecting visual tokens into LLM input) limits deep cross-modal understanding.

## Core Idea

Introduce a visual expert module with separate FFN parameters for visual tokens within each LLM layer, enabling deep fusion while maintaining language capability.

## Architecture

### Overview

![cogvlm2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 512, 512] |
| 2 | Text | `input` |  |
| 3 | EVA-CLIP ViT | `vit` | 64 layers |
| 4 | Visual Projection | `linear` |  |
| 5 | LLaMA-3 LLM | `transformer` | 32 layers |
| 6 | Text Output | `output` |  |

</details>

### Components

1. **EVA-CLIP ViT** — High-capacity vision encoder. 2. **Visual projection** — Maps to LLM space. 3. **Visual expert LLM** — Each transformer layer has separate FFN for visual tokens alongside the language FFN. 4. **Adaptive visual expert** — Expert activated for visual tokens only.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Introduce a visual expert module with separate FFN parameters for visual tokens within each LLM layer, enabling deep fus
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: CogVLM. Successor: CogAgent.

## References

Hong et al. 2024
