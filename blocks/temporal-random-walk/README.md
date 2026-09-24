# Temporal Random Walk Sampler

## Design Philosophy

Static random walks scramble time — a walk may jump backwards through the event history. CTNE constrains walks to *non-decreasing timestamps*, with biased initial-edge selection (exponential/linear recency) and biased temporal-neighbor selection via a monotone decreasing distribution F_T — so the skip-gram context is temporally coherent.

## Functionality

- Walk over continuous-time graph G = (V, E_T, T): next edge sampled among edges with timestamp ≥ current, weighted by recency bias.
- Walks feed a DeepWalk-style Skip-Gram objective → time-dependent node embeddings.

## Used By

| Model | Role |
|-------|------|
| CTNE | Continuous-time dynamic network embeddings |

## Features

- **Causality-respecting contexts** — embeddings reflect "who interacts with whom lately".
- **Recency knobs** — F_T shape and initial-edge bias control the memory horizon.

## Evolution

- **Predecessor**: DeepWalk/node2vec static walks; CTDNE (Nguyen 2018).
- **Related**: metapath-random-walk (heterogeneous type constraints); hawkes-process-embedding (point-process alternative).
