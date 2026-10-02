# Architecture: ALBEF

## Motivation

CLIP aligns image and text but lacks fusion. ViLBERT fuses but lacks alignment. ALBEF bridges this: first align image and text representations with contrastive learning, then fuse them with cross-attention.

## Core Idea

Image and text are encoded separately. A momentum contrastive loss aligns them in a shared space. Then a multimodal encoder fuses them with cross-attention. Momentum distillation provides soft targets.

## Architecture

### Overview

![albef architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3x224x224 |
| 2 | Vision Encoder | `conv2d` | ViT |
| 3 | Text Encoder | `linear` | BERT |
| 4 | Contrastive Alignment | `identity` | ITC loss |
| 5 | Multimodal Encoder | `linear` | cross-attention fusion |
| 6 | VL Output | `output` | ITM + MLM |

</details>

### Components

1. **Image encoder** — ViT extracts image features. 2. **Text encoder** — BERT extracts text features. 3. **Contrastive alignment** — ITC loss aligns image and text embeddings. 4. **Multimodal encoder** — cross-attention fuses image and text. 5. **Momentum model** — EMA-updated teacher provides soft targets. 6. **ITM + MLM** — image-text matching and masked language modeling.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Image and text are encoded separately. A momentum contrastive loss aligns them in a shared space. Then a multimodal enco...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: CLIP (alignment only), ViLBERT (fusion only). Successor: BLIP, CoCa.

## References

- Li et al. 2021
