# Hypersphere SLERP Residual Update (nGPT)

## Design Philosophy

Additive residuals let the hidden state wander off-manifold as depth grows. nGPT constrains *all* embeddings and weights to the unit hypersphere, and each layer updates the state as a spherical interpolation: `h ← Norm(h + α(h_block − h))` — a normalized displacement with learnable per-layer eigen learning rates αA/αM.

## Functionality

- Everything (embeddings, weights) L2-normalized; batch-wise renormalization after each update.
- Learnable logit scale sz; αA (attention), αM (MLP) eigen learning rates per layer; 4–20× faster convergence reported.

## Used By

| Model | Role |
|-------|------|
| nGPT | Per-layer normalized residual update replacing additive residuals |

## Features

- **Bounded state space** — no activation explosion, no drift off-sphere.
- **Learnable step size per layer** — the residual "learning rate" is trained.

## Evolution

- **Predecessor**: additive residuals + pre-norm (the standard stream); spherical-text-embedding geometry.
- **Related**: residual-scaling-init — stability via init/scaling instead of geometry.
