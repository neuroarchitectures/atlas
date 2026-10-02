# Architecture: IP-Adapter

## Motivation

Text-to-image diffusion models are controlled by text. Adding image-based control typically requires fine-tuning the entire model.

## Core Idea

Add a separate cross-attention layer for image features alongside the existing text cross-attention, with a lightweight projection from CLIP image embeddings. Only the new layers are trained.

## Architecture

### Overview

![ip-adapter architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Latent | `input` | shape: [4, 64, 64] |
| 2 | Image Prompt | `input` | shape: [3, 224, 224] |
| 3 | CLIP Image Encoder | `clip-image` |  |
| 4 | Image Projection | `linear` |  |
| 5 | Image Cross-Attention | `cross-attention` |  |
| 6 | UNet (Frozen) | `unet` |  |
| 7 | Generated Image | `output` |  |

</details>

### Components

1. **CLIP image encoder** — Extracts image prompt features. 2. **Image projection** — Maps CLIP features to diffusion model dimension. 3. **Decoupled cross-attention** — Separate cross-attention for image features, added to each UNet block. 4. **Frozen UNet** — Original model weights frozen, only adapter trained.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Add a separate cross-attention layer for image features alongside the existing text cross-attention, with a lightweight 
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ControlNet. Successor: IP-Adapter-Plus, FastComposer.

## References

Ye et al. 2023
