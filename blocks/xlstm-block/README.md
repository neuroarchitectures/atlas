# xLSTM Block (sLSTM / mLSTM)

## Design Philosophy

Extend the LSTM with **exponential gating** (gates are always positive, enabling revision of storage decisions) and two memory variants: **sLSTM** (scalar memory, with memory mixing, sequential) and **mLSTM** (matrix memory, fully parallelizable). The philosophy: address the LSTM's three limitations — inability to revise, limited storage, lack of parallelizability — while keeping the constant-error-carousel principle.

## Functionality

**Exponential gating**: `f_t = exp(f̃_t)`, `i_t = exp(ĩ_t)` — always positive, so gates can fully open and the model can revise past storage.

**sLSTM** (scalar):
- `c_t = f_t ⊙ c_{t-1} + i_t ⊙ z_t` (scalar cell, CEC preserved)
- `n_t = f_t ⊙ n_{t-1} + i_t` (normalizer state for stability)
- `h_t = o_t ⊙ (c_t / max(|n_t|, 1))`
- New memory mixing across cells within a head.

**mLSTM** (matrix):
- `C_t = f_t · C_{t-1} + i_t · v_t k_t^T` (matrix memory, outer-product update)
- `n_t = f_t · n_{t-1} + i_t`
- `h_t = o_t ⊙ (C_t · q_t) / max(|n_t|, 1)` (query-based retrieval)
- No hidden-hidden connections → fully parallelizable.

**Block**: sLSTM/mLSTM with up/down projections, a gated MLP, and (for mLSTM) a conv for local context.

## Used By

| Model | Role |
|-------|------|
| xLSTM | The defining architecture, scaled to billions |

## Features

- **Exponential gates**: Enable revision (overcome LSTM limitation i).
- **Matrix memory (mLSTM)**: d×d storage, addresses rare-token limitation (ii).
- **Parallelizable (mLSTM)**: No hidden-hidden connections (iii).
- **Normalizer state**: Stabilizes exponential dynamics.

## Evolution

- **Predecessor**: LSTM (1997); GRU; Transformer (parallel, outpaced LSTM at scale).
- **Successor**: Ongoing; potential hybrid xLSTM-Transformer architectures; competes with Mamba.
