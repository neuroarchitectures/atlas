# Bidirectional Selective SSM (Vim)

## Design Philosophy

Causal SSMs only see the past — wrong for vision, where context is spatial. Run *two* selective scans, forward and backward over the flattened sequence, and combine: every token receives both-side context, giving a global receptive field at linear complexity, attention-free.

## Functionality

- Per direction d: `y_d = SSMD(x)`; output `y = Δ_d ⊙ y_fwd + Δ_b ⊙ y_bwd` with learned per-direction gates.
- Position embeddings remain mandatory (the scan is order-sensitive); stacked bidirectional blocks form the backbone.

## Used By

| Model | Role |
|-------|------|
| Vim (Vision Mamba) | Stacked bidirectional Mamba blocks over patch sequences |

## Features

- **Linear-complexity global receptive field** — the SSM answer to ViT attention.
- **Learned direction weighting** — the merge is adaptive, not a fixed mean.

## Evolution

- **Predecessor**: S4ND/BiS4 (non-selective bidirectional SSMs).
- **Successor**: ss2d-selective-scan (VMamba) scans in 2D routes instead of two flat directions.
