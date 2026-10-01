# Architecture: MAE

## Motivation

BERT-style masked prediction was successful in NLP but not in vision. MAE shows that masking a high proportion (75%) of image patches and reconstructing them enables strong self-supervised pretraining.

## Core Idea

Mask 75% of image patches. Encoder processes only the 25% visible patches (efficient). Decoder reconstructs the masked patches from the encoded visible patches + mask tokens.

## Architecture

### Overview

![mae architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image Patches | `input` | 196 patches x 768 dim |
| 2 | Mask 75% | `identity` | keep 25% visible, rest masked |
| 3 | ViT Encoder (visible) | `linear` | process only visible patches |
| 4 | Decoder | `linear` | lightweight, reconstruct masked patches |
| 5 | Reconstruction | `output` | MSE loss on masked patches |

</details>

### Components

1. **High masking ratio** — 75% of patches are masked, much higher than BERT's 15%. Vision is highly redundant, so this is feasible. 2. **Asymmetric encoder-decoder** — Encoder is large (ViT) and processes only visible patches. Decoder is small and processes all patches (visible + mask tokens). 3. **Mask token** — A single learnable mask token is inserted for masked positions. 4. **Pretraining** — After pretraining, the decoder is discarded and the encoder is fine-tuned for downstream tasks.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Mask 75% of image patches. Encoder processes only the 25% visible patches (efficient). Decoder reconstructs the masked patches from the encoded visible patches + mask tokens.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: BERT (NLP), BEiT. Successor: SimMIM, CAE, data2vec.

## References

- He et al. 2022
