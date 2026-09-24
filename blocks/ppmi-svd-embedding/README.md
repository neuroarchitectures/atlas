# k-Step PPMI-SVD Embedding

## Design Philosophy

A network's proximities at different walk lengths describe different roles — direct friends vs same-community membership. Compute *shifted PMI matrices per step k* of the adjacency, factorize each by truncated SVD into its own subspace, and concatenate across k: explicit, deterministic multi-scale embedding with no training loop.

## Functionality

- A → D⁻¹W; PPMI_k = PPMI(A^k) for k = 1..K; SVD → d-dim per step (d = 50–200).
- Concatenate across k → |V|×(K·d) embedding.

## Used By

| Model | Role |
|-------|------|
| GraRep | Multi-scale deterministic embeddings |

## Features

- **Training-free** — closed-form spectral factorization.
- **Scale-separated subspaces** — each k has its own coordinates.

## Evolution

- **Predecessor**: SVD on single-scale PMI (Levy & Goldberg word embeddings).
- **Related**: HOPE (asymmetric proximity SVD); high-order-proximity-update (algebraic high-order combination).
