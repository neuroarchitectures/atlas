# Star Operation Block

## Design Philosophy

Element-wise multiplication of two linearly-projected branch activations — the "star operation" — maps features into an implicit high-dimensional non-linear space, like a polynomial kernel: one multiplication yields `(d/√2)²` independent dimensions, and stacking multiplies the effect exponentially. A near-free alternative to widening channels.

## Functionality

- `U = DWConv(W1 X)`, `V = DWConv(W2 X)`; output `U ⊙ V` (± fusion convs).
- Stacked star blocks reach convnext-like accuracy at compact width, without channel widening.

## Used By

| Model | Role |
|-------|------|
| StarNet | Compact backbone of stacked star blocks, no channel widening |

## Features

- **Exponential feature-space growth** per stacked multiplication — the paper's central analysis.
- **Minimal latency** — one extra pointwise + depthwise per block.

## Evolution

- **Predecessor**: high-dimensional MLP expansions; SE-style channel gating (squeeze-and-excite).
- **Related**: gated activations (SiLU-gated convs), HorNet's gnConv (higher-order multiplicative interactions).
