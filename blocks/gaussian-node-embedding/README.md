# Gaussian Node Embedding

## Design Philosophy

A node's position in a network is uncertain — represent it as a *distribution*: each node is a Gaussian N(μ, Σ). Similarity becomes a closed-form Wasserstein or expected-likelihood distance, uncertainty is first-class, and DVNE additionally couples mean and variance through the variational objective so transitivity is preserved by the true metric.

## Functionality

- DVNE: d-dim Gaussians, reparameterization z = μ + σε; 2-Wasserstein similarity; first+second-order proximity losses, negative sampling.
- SDGE variant: MLP with separate mean/log-variance heads → diagonal covariance; inductive forward pass for unseen nodes.

## Used By

| Model | Role |
|-------|------|
| DVNE | Deep variational network embedding in Wasserstein space |
| SDGE | Stochastic deep graph embedding with hop-distance supervision |

## Features

- **Uncertainty-aware similarity** — ambiguous nodes sit near several clusters.
- **Metric properties** (DVNE) — Wasserstein distance preserves transitivity.

## Evolution

- **Predecessor**: point embeddings (node2vec); DANCOSE probabilistic embeddings.
- **Related**: poincare-embedding — the hyperbolic alternative for hierarchy; poincare-style variational distance objectives (SDGE's hop-distance-ranking-loss).
