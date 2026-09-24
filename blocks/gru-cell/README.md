# Gated Recurrent Unit (GRU)

## Design Philosophy

LSTM's two states and three gates are redundant for many tasks. The GRU collapses to one hidden state with two gates — update (how much to overwrite) and reset (how much history to expose) — keeping the gating benefits at a third fewer parameters.

## Functionality

- `z_t = σ(W_z[h_{t-1}, x_t])`, `r_t = σ(W_r[h_{t-1}, x_t])`, `h̃_t = tanh(W[r_t ⊙ h_{t-1}, x_t])`, `h_t = (1-z_t)h_{t-1} + z_t h̃_t`.
- Sequence-side: interest evolution over target-attended histories; vision-side: the convolutional variant (convgru-update-operator) refines dense fields.

## Used By

| Model | Role |
|-------|------|
| DIEN | Interest-evolution GRU (AUGRU variant) over target-attended behavior sequence |
| SEA-RAFT | Per-pixel GRU iterating flow residuals over correlation lookups |

## Features

- **Fewer gates, comparable quality** — the pragmatic recurrent cell.
- **Parallelizable variants** (minGRU/minLSTM) revived it in the linear-attention era.

## Evolution

- **Predecessor**: lstm-cell (simplification of).
- **Successor**: selective-ssm and linear-attention cells replace gating with state-space dynamics for long contexts.
