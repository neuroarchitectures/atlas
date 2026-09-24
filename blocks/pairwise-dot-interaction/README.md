# Pairwise Dot-Interaction

## Design Philosophy

Feature interactions in CTR prediction are dominated by pairwise products. DLRM computes *all pairwise dot products* between sparse embeddings (and the dense vector), feeding the upper-triangular interactions to the top MLP — explicit second-order interactions absorbed into a deep net.

## Functionality

- 26 sparse embeddings (dim 32) + dense 32-d processed vector → all pairwise dots → upper triangle → top MLP (415→512→256).

## Used By

| Model | Role |
|-------|------|
| DLRM | Interaction layer between bottom MLP and top MLP; the recsys throughput benchmark architecture |

## Features

- **Every pair considered** — no feature-pair selection heuristic.
- **Quadratic but tiny** — 32-dim dots are cheap; the reference scale of recsys compute.

## Evolution

- **Predecessor**: FM (factorization machines), FFM.
- **Related**: deep-cross-network (learned crosses), gmf-mlp-fusion (product + MLP paths).
