# Architecture: BEiT

## Motivation

MAE reconstructs raw pixels, but vision is less redundant than assumed. BEiT predicts discrete visual tokens (from a VQ-VAE) instead, similar to BERT predicting word tokens. This provides a more semantic pretraining signal.

## Core Idea

Image patches are tokenized using a pretrained VQ-VAE (dVAE). 40% of patches are masked. A ViT encoder processes visible patches. A classification head predicts the discrete visual tokens of masked patches.

## Architecture

### Overview

![beqv2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image Patches | `input` | 14x14 patches |
| 2 | Mask 40% | `identity` | random masking |
| 3 | ViT Encoder | `linear` | 12 layers, 768 dim |
| 4 | Classification Head | `linear` | predict VQ tokens |
| 5 | Visual Token Prediction | `output` | 8192 codebook entries |

</details>

### Components

1. **VQ-VAE tokenizer** — pretrained dVAE converts patches to discrete tokens. 2. **Block-wise masking** — 40% of patches masked (vs 75% in MAE). 3. **ViT encoder** — standard transformer encoder on visible patches. 4. **Cross-entropy loss** — predicts discrete visual tokens, not pixels. 5. **Fine-tuning** — encoder fine-tuned for downstream tasks.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Image patches are tokenized using a pretrained VQ-VAE (dVAE). 40% of patches are masked. A ViT encoder processes visible...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: MAE (pixel reconstruction), BERT (NLP). Successor: BEiT v2, data2vec.

## References

- Bao et al. 2021
