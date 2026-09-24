# GMF + MLP Fusion

## Design Philosophy

Matrix factorization's element-wise product captures linear interactions; an MLP over concatenated embeddings captures non-linear ones — neither dominates. NeuMF runs *both* with separate embedding tables and fuses their outputs before the score head, letting each path specialize.

## Functionality

- GMF path: `e^U_MF ⊙ e^I_MF` (element-wise product, 32-dim).
- MLP path: concat(e^U_MLP, e^I_MLP) → stacked MLP (64-dim embeddings); final score from fused `α·GMF + (1-α)·MLP` (α learned).

## Used By

| Model | Role |
|-------|------|
| NeuMF | Collaborative filtering with two specialized interaction paths |

## Features

- **Complementary inductive biases** — linear + non-linear interaction channels.
- **Separate embeddings** — pretraining (MF) and end-to-end (MLP) modes coexist.

## Evolution

- **Predecessor**: pure MF; pure MLP CF (NCF's MLP-only instantiation).
- **Related**: pairwise-dot-interaction (DLRM) — dot products feeding a deeper MLP.
