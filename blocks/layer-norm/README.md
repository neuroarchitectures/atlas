# LayerNorm (Layer Normalization)

## Design Philosophy

Normalize **across the feature dimension for each sample**, not across the batch. The motivation is to break the dependence on batch statistics: BatchNorm's behavior changes with batch size and is awkward for variable-length sequences, so LayerNorm computes mean and variance from the sample itself. It is the default normalization for transformers.

## Functionality

`LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β`, where `μ` and `σ²` are the mean and variance over the normalized axes (usually the hidden dim, per token), and `γ, β` are learnable per-channel affine parameters.

- Statistics are computed **per sample**, so train and inference behave identically.
- Independent of batch size and sequence length.
- Two placements: **post-norm** (`LN(x + Sublayer(x))`, the original transformer) and **pre-norm** (`x + Sublayer(LN(x))`, the modern default for deep stacks).

## Used By

| Model | Role |
|-------|------|
| Transformer / BERT | Post-norm after attention and FFN |
| GPT-2 / ViT / Swin | Pre-norm (stabilizes deeper stacks) |
| Diffusion Transformer (DiT) | Base for adaptive LayerNorm (adaLN) conditioning |
| DETR / RT-DETR | Norm inside transformer encoder/decoder blocks |

## Features

- **Batch-independent**: same computation at train and inference time.
- **Pre-norm** enables very deep transformers without warmup tricks; **post-norm** gives slightly better final quality when it does converge.
- Cheap, but requires one full reduction over the hidden dim.

## Evolution

- **Predecessor**: BatchNorm (Ioffe & Szegedy, 2015).
- **Itself**: LayerNorm (Ba et al., 2016).
- **Variants/successors**: RMSNorm (drops mean subtraction), Adaptive LayerNorm / adaLN-Zero (scale-shift from conditioning), ScaleNorm, DeepNorm (residual scaling for 1000+ layers).
