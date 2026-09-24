# Variance-Reduced GCN Sampling

## Design Philosophy

Full-batch GCN doesn't scale; naive neighbor sampling has unbounded variance. Two fixes from the same goal: layer-wise *importance sampling* of vertices with q(u) ∝ ‖Â(:,u)‖² (FastGCN, sum not product of sample sizes), and *control variates from historical activations* (Stochastic-GCN) — cache per-node per-layer activations, apply Monte-Carlo only to the residual, so variance vanishes as training converges.

## Functionality

- FastGCN: per-layer vertex sampling with importance weights; precomputed ÂH⁽⁰⁾ for stability.
- Stochastic-GCN: `ĥ = hist + MC-sampled(Δh)`; history refreshed each epoch; enables D(l)=2 samples with convergence guarantees; CVD variant for dropout, PP preprocessing (PX).

## Used By

| Model | Role |
|-------|------|
| FastGCN | 2 gcn_conv layers, t_l vertices sampled per layer |
| Stochastic-GCN | Per-node activation cache + MC residual sampling |

## Features

- **Convergence-guaranteed subsampling** — not just unbiased, but vanishing-variance at optimum.
- **Scalable to large graphs** with small per-layer sample budgets.

## Evolution

- **Predecessor**: GraphSAGE neighbor sampling, GraphSAINT.
- **Successor**: cluster-GCN; decoupled propagation (SGC) that removes sampling entirely.
