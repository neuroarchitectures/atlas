# Masked Autoencoder (MAE)

## Design Philosophy

Mask most of the input, encode only the visible tokens, and reconstruct what's missing with a lightweight decoder. Asymmetric masking makes pretraining ~3× faster and forces the encoder — not the decoder — to own the representation. The *target* of reconstruction is the design knob: pixels, features, or discrete IDs.

## Functionality

- Random masking (75%+ for images; tubes for video) → ViT encoder over visible tokens → lightweight decoder reconstructs targets.
- Target variants: pixels (VideoMAE), teacher image embedding (EfficientSAM's SAMI — decoder discarded after pretraining), context pixels (SimMIM trains end-to-end without decoder), discrete cluster IDs (HuBERT's masked-unit prediction).

## Used By

| Model | Role |
|-------|------|
| VideoMAE | Pixel reconstruction over 3D tube tokens |
| EfficientSAM | SAMI: reconstruct SAM ViT-H's image embedding; then SAM decoder fine-tune |
| Swin-V2 (SimMIM) | Masked-pixel pre-training of the Swin encoder (~40× label reduction) |
| HuBERT | Masked cluster-ID prediction from offline k-means units |

## Features

- **Encoder-centric representation** — decoder is scaffolding, dropped at fine-tune.
- **High mask ratio tolerable** — cheap and strongly regularizing.

## Evolution

- **Predecessor**: BERT masked-language-modeling; iGPT.
- **Successor**: masked-latent-prediction (JEPA family) — predict target *representations*, not inputs.
