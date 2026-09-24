# DeepFM Block

## Design Philosophy

Fuse a **factorization machine (FM)** with a deep network under one shared embedding table. The philosophy: the FM branch captures explicit second-order (pairwise) feature interactions; the deep MLP captures higher-order interactions implicitly; both read the *same* embeddings, so neither path needs hand-engineered crosses (unlike Wide & Deep).

## Functionality

- **Shared embeddings**: Sparse features → one embedding table, shared by both branches.
- **FM branch**: Explicit second-order interactions: `Σ_{i<j} <v_i, v_j> x_i x_j` — pairwise inner products of embeddings.
- **Deep branch**: Embeddings flattened → MLP (FC-ReLU-...-FC).
- **Fusion**: `logit = FM_output + deep_output` → sigmoid.

## Used By

| Model | Role |
|-------|------|
| DeepFM (Huawei) | CTR / click prediction |
| xDeepFM | Adds compressed interaction network to FM |
| AutoInt | Adds self-attention for feature interactions |

## Features

- **No manual crosses**: The FM learns pairwise interactions automatically.
- **Shared embeddings**: Both branches co-train, no separate feature engineering.
- **Low-order + high-order**: FM = 2nd-order, MLP = higher-order.

## Evolution

- **Predecessor**: Wide & Deep (needs manual crosses); FM (Factorization Machine, no deep path).
- **Successor**: xDeepFM, AutoInt, DCN — the family of "learned interaction" CTR models.
