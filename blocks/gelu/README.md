# GELU (Gaussian Error Linear Unit)

## Design Philosophy

Instead of a hand-designed gate like ReLU's hard zero, GELU multiplies the input by **the probability that a standard Gaussian would take a value below it**: `x · Φ(x)`. Activations are therefore weighted by their quantile rather than dropped, giving a smooth, non-monotonic function that empirically outperforms ReLU and ELU in transformer FFNs.

## Functionality

`GELU(x) = x · P(X ≤ x) = x · Φ(x)`, with `X ~ N(0, 1)`.

- Exact form uses the error function; in practice the **tanh approximation** or the sigmoid approximation `x · σ(1.702 x)` is used for speed.
- Smooth everywhere (unlike ReLU at 0), with a small negative dip around `x ≈ −0.75`, so a few negative values pass through.
- Drop-in replacement for ReLU with no new hyperparameters.

## Used By

| Model | Role |
|-------|------|
| BERT / RoBERTa / ALBERT | FFN activation |
| GPT-2 / GPT-3 | FFN activation |
| Vision Transformer (ViT), DINOv2, Depth Anything V2 | FFN activation |
| DETR / Deformable DETR transformer blocks | FFN activation |

## Features

- **Smooth and non-monotonic**: slightly better gradient flow than ReLU.
- **Stochastic-regularizer interpretation**: `x · Φ(x)` equals the expected output of a stochastic gate, linking it to dropout and zoneout.
- Cost: exact GELU needs `erf`, so the tanh/sigmoid approximation matters for throughput.

## Evolution

- **Predecessors**: sigmoid/tanh (saturating), ReLU (2010s default), LeakyReLU / ELU.
- **Itself**: GELU (Hendrycks & Gimpel, 2016).
- **Successors/competitors**: Swish/SiLU (self-gating, discovered by NAS — same shape family), and gated units SwiGLU / GeGLU which use GELU/SiLU as the gate in an MLP.
