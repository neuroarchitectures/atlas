# Q-Former (Learned Query Bridge)

## Design Philosophy

Bridge a frozen image encoder and a frozen LLM with a lightweight **Querying Transformer**: a fixed set of learned query tokens cross-attend the image features, distilling them into a fixed-length visual summary the LLM can read. The philosophy: the vision encoder and LLM are both expensive pretrained models — train only the tiny bridge (the Q-Former), so almost no parameters are trained.

## Functionality

- **Learned queries**: 32 (or 64) trainable query tokens.
- **Self-attention**: Queries attend to each other.
- **Cross-attention**: Queries cross-attend the frozen image features (ViT output).
- **Output**: The 32 query vectors → linear projection → LLM token space.
- **Training**: Only the Q-Former (and projection) are trained; ViT and LLM stay frozen.

## Used By

| Model | Role |
|-------|------|
| BLIP-2 | The defining bridge (Q-Former) |
- Contrast with LLaVA's simple MLP projector (direct projection, no cross-attention).

## Features

- **Fixed output length**: 32 visual tokens regardless of image size.
- **Compressed**: Cross-attention distills variable-length image features to a constant summary.
- **Cheap to train**: Only the Q-Former (+ projection) — ViT and LLM frozen.

## Evolution

- **Predecessor**: Perceiver (learned latents cross-attend inputs); Flamingo's Perceiver Resampler.
- **Successor**: LLaVA's MLP projector (simpler, no cross-attention); the "bridge" pattern is now standard in MLLMs.
