# Architecture: BLIP

## Motivation

Vision-language models are either good at understanding (VQA) or generation (captioning), but not both. Noisy web data degrades performance.

## Core Idea

Use a shared multimodal encoder-decoder with three objectives: image-text contrastive learning, image-text matching, and image-grounded text generation. Bootstraps clean captions from noisy web data using a captioning model as data cleaner.

## Architecture

### Overview

![blip architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 384, 384] |
| 2 | Text | `input` |  |
| 3 | Vision Encoder | `vit` | 12 layers |
| 4 | Text Encoder | `bert` | 12 layers |
| 5 | Cross-Attention | `cross-attention` |  |
| 6 | LM Head | `lm-head` |  |
| 7 | Text / Alignment | `output` |  |

</details>

### Components

1. **Vision encoder** — ViT extracts image features. 2. **Text encoder** — BERT-like text encoder. 3. **Cross-attention** — Fuses image and text features. 4. **LM head** — Autoregressive generation for captioning. 5. **Bootstrapping** — Self-generated captions filter noisy web pairs.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use a shared multimodal encoder-decoder with three objectives: image-text contrastive learning, image-text matching, and
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: CLIP, ALBEF. Successor: BLIP-2, InstructBLIP.

## References

Li et al. 2022
