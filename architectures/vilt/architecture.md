# Architecture: ViLT

## Motivation

Vision-language models use heavy visual backbones (ResNet, ViT) or region features (Faster R-CNN). ViLT shows that simple linear patch embeddings (no CNN) are sufficient when combined with a transformer, simplifying the pipeline.

## Core Idea

Image patches are embedded with a simple linear projection (no CNN backbone). Text tokens are embedded normally. Both are concatenated with type embeddings and passed through a single transformer encoder.

## Architecture

### Overview

![vilt architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3x224x224 |
| 2 | Patch Embedding | `identity` | linear projection |
| 3 | Text Embedding | `linear` | word + position |
| 4 | Concat + Type Embed | `identity` | [CLS] img patches text |
| 5 | Transformer Encoder | `linear` | 12 layers |
| 6 | VL Output | `output` | ITM + MLM |

</details>

### Components

1. **Patch embedding** — linear projection of image patches (no CNN). 2. **Text embedding** — standard word + position embeddings. 3. **Type embedding** — distinguishes image vs text tokens. 4. **Single transformer** — processes both modalities jointly. 5. **No region features** — no object detector needed. 6. **Pretraining** — ITC, ITM, and MLM losses.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Image patches are embedded with a simple linear projection (no CNN backbone). Text tokens are embedded normally. Both ar...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: CLIP, ViLBERT. Successor: ALBEF, BLIP.

## References

- Kim et al. 2021
