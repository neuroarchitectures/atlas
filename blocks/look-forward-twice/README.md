# Look-Forward-Twice Box Supervision

## Design Philosophy

In an iterative box-refinement decoder, each layer is greedy: it optimizes its own output box, unaware that a later layer will refine it again. Look-forward-twice supervises each layer's box delta *also from the perspective of the next layer's refined box*, aligning layers toward the final result instead of local optima.

## Functionality

- Layer ℓ's delta d_ℓ is applied relative to layer ℓ+1's refined box: `b_{ℓ+1} = b_ℓ + d_ℓ(b_ℓ)` evaluated twice — once for layer ℓ's loss, once inside layer ℓ+1's.
- Applied across the decoder's box head; training-only formulation change (no extra params).

## Used By

| Model | Role |
|-------|------|
| DINO | Box supervision across the 6 decoder layers |

## Features

- **Greedy-lock-in fix** for any iterative refinement decoder — no architecture change.
- **Cheap** — a reformulation of the loss perspective.

## Evolution

- **Predecessor**: iterative box refinement (Deformable DETR).
- **Related**: D-FINE's go-lsd-self-distillation — the same "teach early layers from late layers" idea via distillation of distributions.
