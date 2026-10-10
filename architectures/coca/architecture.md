# Architecture: CoCa

## Motivation

CLIP uses only contrastive learning (global alignment). ALBEF uses contrastive + matching. CoCa adds captioning (generation) for richer understanding, achieving state-of-the-art on VQA and retrieval.

## Core Idea

Vision encoder produces patch tokens. Text encoder produces token embeddings. A contrastive head aligns global features (CLIP-style). A captioning head generates text from image patches (generative).

## Architecture

### Overview

![coca architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | ViT Encoder | `conv` | patch 16, 768 dim |
| 3 | Attention Pooling | `attention` | single-token attention pool |
| 3 | Captioning Decoder | `attention` | cross-attention to image patches |
| 4 | Contrastive Head | `linear` | CLIP-style alignment |
| 4 | Captioning Loss | `output` | autoregressive text generation |
| 5 | Contrastive Loss | `output` | InfoNCE |

</details>

### Components

1. **ViT encoder** — Standard vision transformer producing patch-level features. 2. **Attention pooling** — A single attention pool token aggregates patch features into a global representation for contrastive learning. 3. **Captioning decoder** — A transformer decoder that cross-attends to image patches and generates text autoregressively. 4. **Dual loss** — Contrastive loss (CLIP-style) + captioning loss (cross-entropy). 5. **No bounding boxes** — Unlike ALBEF, CoCa doesn't use region labels.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Vision encoder produces patch tokens. Text encoder produces token embeddings. A contrastive head aligns global features (CLIP-style). A captioning head generates text from image patches (generative).
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: CLIP, ALBEF. Successor: PaLI, PaLI-2.

## References

- Yu et al. 2022
