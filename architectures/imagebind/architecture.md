# Architecture: ImageBind

## Motivation

CLIP aligns image and text. ImageBind extends this to 6 modalities (image, text, audio, depth, thermal, IMU) by binding them all to image embeddings, using only image-paired data.

## Core Idea

Each modality has its own encoder. All encoders are trained to align their outputs to the image encoder's output space. Only image-paired data is needed; other modality pairs emerge naturally.

## Architecture

### Overview

![imagebind architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Any Modality | `input` | image/text/audio/depth/thermal/IMU |
| 2 | Modality Encoder | `linear` | modality-specific encoder -> 1024 dim |
| 3 | LayerNorm | `layerNorm` |  |
| 4 | Shared Embedding | `output` | aligned to image space |

</details>

### Components

1. **Modality-specific encoders** — Each modality (image: ViT, text: Transformer, audio: AudioSpectrogramTransformer, etc.) has its own encoder. 2. **Image as anchor** — All encoders are trained to align to the image encoder's output. Only image-paired data is needed for each modality. 3. **Emergent cross-modal alignment** — Since all modalities align to image, they align to each other automatically. 4. **Contrastive loss** — InfoNCE loss between modality pairs and image pairs.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Each modality has its own encoder. All encoders are trained to align their outputs to the image encoder's output space. Only image-paired data is needed; other modality pairs emerge naturally.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: CLIP, AudioCLIP. Successor: ImageBind variants, 6-modal foundation models.

## References

- Girdhar et al. 2023
