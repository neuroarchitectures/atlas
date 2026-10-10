# Architecture: DALL-E 2

## Motivation

CLIP aligns text and image but can't generate images. DALL-E 2 bridges this gap: a diffusion prior maps CLIP text embeddings to CLIP image embeddings, then a decoder generates the image from that embedding.

## Core Idea

CLIP text encoder produces text embeddings. A diffusion prior (autoregressive or diffusion) maps text embeddings to CLIP image embeddings. A GLIDE-based decoder generates the image conditioned on the CLIP image embedding + text. Super-resolution upscales.

## Architecture

### Overview

![dalle-2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Text Prompt | `input` | text |
| 2 | CLIP Text Encoder | `linear` | frozen CLIP |
| 3 | Diffusion Prior | `linear` | text emb -> image emb |
| 4 | GLIDE Decoder | `conv2d` | image emb -> 64x64 |
| 5 | Super-Resolution | `conv2d` | 64 -> 256 |
| 6 | Generated Image | `output` | 256x256 |

</details>

### Components

1. **CLIP text encoder** — frozen, produces text embeddings. 2. **Diffusion prior** — maps text embedding to image embedding space. 3. **GLIDE decoder** — diffusion model conditioned on CLIP image embedding. 4. **Super-resolution** — upsamples to higher resolution. 5. **Two-stage** — prior + decoder, both diffusion-based. 6. **CLIP embeddings** — bridge between text and image.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — CLIP text encoder produces text embeddings. A diffusion prior (autoregressive or diffusion) maps text embeddings to CLIP...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DALL-E (autoregressive), CLIP, GLIDE. Successor: Imagen, Stable Diffusion.

## References

- Ramesh et al. 2022
