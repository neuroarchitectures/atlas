# Target Attention (DIN Activation Unit)

## Design Philosophy

Replace fixed sum-pooling of a user's behavior history with an attention "activation unit": each past behavior is **weighted by how relevant it is to the candidate item**, so the user's interest vector is target-aware. The philosophy: a user's interest is multi-modal — when scoring a candidate, attend to the part of the history relevant to *that* candidate.

## Functionality

- **Inputs**: Candidate item embedding `c`, historical behavior embeddings `{b_1, ..., b_T}`.
- **Activation unit**: For each behavior `b_t`, compute attention weight `α_t = MLP([b_t, c, b_t - c, b_t ⊙ c])` → softmax over `t`.
- **Interest vector**: `interest = Σ_t α_t · b_t` — relevance-weighted sum (not mean pool).
- **Head**: Concat(interest, candidate) → MLP → sigmoid.

## Used By

| Model | Role |
|-------|------|
| DIN (Deep Interest Network, Alibaba) | Production CTR with target-aware pooling |
| DIEN | Adds a GRU-based interest-evolution layer on top |
| BST | Replaces the activation unit with a full Transformer |

## Features

- **Target-aware**: The interest representation depends on the candidate.
- **Local attention**: A single-head, single-query attention over the behavior sequence.
- **Multi-modal interest**: Different candidates surface different historical behaviors.

## Evolution

- **Predecessor**: Fixed sum/mean pooling of behaviors (Wide & Deep, DeepFM).
- **Successor**: DIEN (GRU interest evolution); BST (Transformer over behaviors); the target-attention pattern is now standard in recsys.
