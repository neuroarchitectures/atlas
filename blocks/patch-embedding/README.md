# Patch Embedding (ViT)

## Design Philosophy

Turn an image into a sequence of tokens so a Transformer can process it. The philosophy: ViT has no inductive bias toward spatial locality, so the patch embedding provides the only "image-aware" operation. A strided 2D conv (or unfold + linear) chops the image into non-overlapping patches and projects each to a token vector. The rest of the model is a plain Transformer.

## Functionality

- **Patchify**: Split a `H×W×C` image into `N = (H/P)·(W/P)` patches of size `P×P×C`.
- **Linear projection**: Each patch (flattened to `P²·C` dims) → linear to `D` (the embedding dim).
- **Implementation**: Equivalent to a `conv2d` with `kernelSize = P`, `stride = P`, `outChannels = D`.
- **CLS token**: A learnable token prepended (ViT) or pooled (ViT variants).
- **Position embedding**: Added on top (learned or RoPE).

## Used By

| Model | Role |
|-------|------|
| ViT-B/16 | 16×16 patches → 196 tokens, 768-dim |
| Swin Transformer | 4×4 patchify conv (96-dim) |
| DiT-XL/2 | 2×2 patches over a 4×32×32 latent |
| CLIP ViT-B/32 | 32×32 patches, 768-dim |
| BEiT, MAE | Same patchify + linear |

## Features

- **Tokenization**: The bridge from pixels to the Transformer's discrete-token world.
- **Sequence length control**: Smaller patches → more tokens → more cost, more detail.
- **No overlap**: Standard ViT uses non-overlapping patches; overlapping variants exist.

## Evolution

- **Predecessor**: Conv stems (ResNet) — the strided conv is the same primitive, but ViT uses it once as a tokenizer, not as a stack.
- **Successor**: ConvStem (early-conv-then-Transformer) for better data efficiency; hierarchical patch merging (Swin) for pyramid features.
