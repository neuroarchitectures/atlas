# Architecture: Imagen

## Motivation

CLIP text encoders have limited language understanding for complex prompts. Imagen uses a large frozen T5 (4.6B params) for superior text comprehension, paired with cascaded diffusion for high-resolution generation.

## Core Idea

A frozen T5-XXL text encoder produces rich text embeddings. A base diffusion model generates 64x64 images. Two super-resolution diffusion models upscale to 256x256 then 1024x1024. Text conditioning via cross-attention at all stages.

## Architecture

### Overview

![imagen architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Text Prompt | `input` | text |
| 2 | T5 Text Encoder | `linear` | frozen T5-XXL, 4.6B |
| 3 | Base Diffusion | `conv2d` | 64x64 |
| 4 | Super-Resolution 1 | `conv2d` | 64 -> 256 |
| 5 | Super-Resolution 2 | `conv2d` | 256 -> 1024 |
| 6 | Generated Image | `output` | 1024x1024 |

</details>

### Components

1. **Frozen T5-XXL** — 4.6B parameter text encoder, not fine-tuned. 2. **Cascaded diffusion** — base (64x64) + two super-resolution stages. 3. **Cross-attention** — text conditioning at every stage. 4. **Classifier-free guidance** — conditional + unconditional. 5. **Dynamic thresholding** — improves sample quality at high guidance. 6. **No VAE** — operates directly in pixel space.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A frozen T5-XXL text encoder produces rich text embeddings. A base diffusion model generates 64x64 images. Two super-res...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DALL-E 2, GLIDE. Successor: Parti (autoregressive), Imagen 2/3.

## References

- Saharia et al. 2022
