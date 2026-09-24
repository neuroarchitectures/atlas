# Masked Attention

## Design Philosophy

DETR-style decoders converge slowly because each query must first discover *where* its object is among all image features. Masked attention restricts each query's cross-attention to the foreground region of the mask it predicted in the previous layer — attention starts focused, and focus refines layer by layer.

## Functionality

- Decoder layer ℓ: query's predicted mask from layer ℓ−1 (thresholded at 0.5) masks encoder features before cross-attention (masked positions get −∞).
- Round-robin assignment of multi-scale feature levels per layer; trained jointly with mask + classification losses.

## Used By

| Model | Role |
|-------|------|
| Mask2Former | Core decoder mechanism over multi-scale pixel-decoder features |

## Features

- **Fast convergence** — the defining fix over DETR/MaskFormer decoders (up to ~10× fewer epochs).
- **Local-by-construction** — each query only sees its own region, keeping queries specialized.

## Evolution

- **Predecessor**: MaskFormer (per-segment classification), DETR cross-attention.
- **Related**: multi-scale-deformable-attention (DINO) — sparse sampling as the alternative convergence fix.
