# Decoupled Cross-Attention (IP-Adapter)

## Design Philosophy

Adding a new condition (e.g. an image prompt) to a frozen text-to-image diffusion model by mixing it into the existing text cross-attention couples the new signal to the text keys/values and risks disturbing the base model. A decoupled cross-attention adds a *separate* attention branch for the new condition and sums it in.

## Functionality

Keep the original text cross-attention and add a parallel one for the image:

```
out = Attn(Q, K_text, V_text) + lambda * Attn(Q, K_image, V_image)
```

The second branch has its own projection weights and is trainable while the base UNet stays frozen; `lambda` (or a learned scalar) controls its strength. The image branch is zero-initialised so training starts from the base behaviour.

## Used By

IP-Adapter

## Features

- **Independent conditioning**: image and text attention do not interfere.
- **Frozen backbone**: only the adapter branch trains.
- **Composable**: multiple adapters (style, layout) can be summed.
- **Zero-init**: starts as the unmodified model and grows in.

## Evolution

Predecessor: cross-attention (single condition), ControlNet (spatial conditioning). Sibling: gated cross-attention (Flamingo), visual expert (CogVLM). Successor: multi-adapter composition in diffusion.
