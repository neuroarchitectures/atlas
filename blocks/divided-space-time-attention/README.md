# Divided Space-Time Attention

## Design Philosophy

Joint space-time attention on video tokens is quadratic in T×H×W. TimeSformer divides: within each block, apply *temporal* attention first (each spatial location across frames), then *spatial* attention (each frame across locations) — near-joint quality at a fraction of the cost, convolution-free.

## Functionality

- Per block: temporal attention (T axis) → residual → spatial attention (HW axis) → residual → MLP.
- Temporal-first ordering validated by ablation; joint attention only as a costly upper bound.

## Used By

| Model | Role |
|-------|------|
| TimeSformer | Frame-level patches, per-block temporal-then-spatial, ViT-style backbone |

## Features

- **O(T·HW + HW·T)** instead of O(T·HW·T·HW) per head.
- **Modality factorization** — temporal and spatial mixing are separately interpretable.

## Evolution

- **Predecessor**: ViT per-frame (no temporal mixing), joint space-time ViT.
- **Successor**: factorised-space-time-encoder (ViViT, separate stacks); joint-spacetime-attention (CoTracker, joint when budget allows); tubelet-embedding handles tokenization.
