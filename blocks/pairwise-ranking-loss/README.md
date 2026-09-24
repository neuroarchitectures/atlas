# Pairwise Ranking Loss (BPR / Margin / Hop-Distance)

## Design Philosophy

Recommendation and embedding tasks don't need absolute scores — only correct *order*. Pairwise ranking losses compare one positive against sampled negatives: BPR/TOP1 over session items (GRU4Rec), margin losses with hard negatives (PinSage), hop-distance ordering (SDGE), structural-balance margins (SNE) — the shared mechanism is maximize(score(pos) − score(neg) + margin), with the sampling strategy as the real design choice.

## Functionality

- BPR: `−log σ(s_pos − s_neg)`; TOP1: hinge over sampled negatives.
- PinSage: margin Δ with 500 shared negatives + 6 PPR-ranked hard negatives per pin, curriculum-increased per epoch.
- SDGE: pairwise ranking on Gaussian dissimilarity enforcing hop k > hop k−1 distance. SNE: balance-theory margins pulling positive-linked pairs together, negative-linked apart.

## Used By

| Model | Role |
|-------|------|
| GRU4Rec | TOP1/BPR over sampled negative items, session-parallel mini-batching |
| PinSage | Margin ranking with PPR hard negatives |
| SDGE | Hop-distance ranking over Gaussian embeddings |
| SNE | Signed-balance margin ranking with attribute-similarity term |

## Features

- **Order-only supervision** — robust to score scale drift.
- **Negative sampling is the lever** — hard negatives dominate quality.

## Evolution

- **Predecessor**: pointwise losses; RankNet (2005).
- **Related**: listwise losses (softmax over candidates), contrastive losses with temperature.
