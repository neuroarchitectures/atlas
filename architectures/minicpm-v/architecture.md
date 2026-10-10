# Architecture: MiniCPM-V

## Motivation

Deploying VLMs on mobile devices requires efficient architectures that maintain multimodal capability at small scale.

## Core Idea

Use visual compression (token reduction) and adaptive embedding to feed compressed visual tokens into a small but effective LLM, enabling on-device multimodal inference.

## Architecture

### Overview

![minicpm-v architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 448, 448] |
| 2 | Text | `input` |  |
| 3 | ViT | `vit` | 26 layers |
| 4 | Visual Compression | `compress` | ratio: 4 |
| 5 | Projection | `linear` |  |
| 6 | MiniCPM LLM | `transformer` | 40 layers |
| 7 | Text Output | `output` |  |

</details>

### Components

1. **ViT encoder** — Vision transformer for image feature extraction. 2. **Visual compression** — Spatial pooling reduces visual token count by 4x. 3. **Projection** — Maps visual features to LLM space. 4. **MiniCPM LLM** — Compact transformer (2-3B params) with strong language capability. 5. **Adaptive embedding** — Visual tokens adaptively scaled.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use visual compression (token reduction) and adaptive embedding to feed compressed visual tokens into a small but effect
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: MiniCPM. Successor: MiniCPM-V 2.6, OmniLMM.

## References

Yao et al. 2024
