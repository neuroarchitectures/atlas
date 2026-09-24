# High-Order Proximity Update

## Design Philosophy

High-order proximity (A^(2k)) is what long-range structure needs, but recomputing embeddings per snapshot is prohibitive. Two closed-form shortcuts from the same insight: NEU algebraically combines low-order embeddings to *approximate* A^(2k) with a provable bound (FastNE), and matrix *perturbation theory* applies a first-order eigenspace correction when the graph changes by ΔA/ΔX (DANE) — both run in <1% of training time.

## Functionality

- NEU: `R_high = Σ w_k · algebraic combination(R_k)` over existing low-order embeddings.
- DANE: consensus embedding (structure + attributes) → incremental perturbation update per timestep, no recomputation.

## Used By

| Model | Role |
|-------|------|
| FastNE | Post-processing on any NRL embedding R ∈ ℝ^(|V|×d) |
| DANE | Online dynamic-network embedding with attribute consensus |

## Features

- **Provable approximation quality** — bounds, not heuristics.
- **Incremental dynamics** — embeddings track graph evolution at stream speed.

## Evolution

- **Predecessor**: recomputing spectral embeddings; offline High-Order Proximity Preserved embedding.
- **Related**: matrix-perturbation streaming ideas in dynamic SVD; temporal-random-walk (the sampling alternative).
