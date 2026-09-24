# RWKV Channel Mixing

## Design Philosophy

The FFN counterpart to RWKV time-mixing: keep the recurrent receptance gating pattern but mix over *channels* with token-shift on the key — a parallelizable gated feed-forward that complements the sequence-mixing sub-block.

## Functionality

- `r_t = σ(W_r · (μ_r x_t + (1-μ_r) x_{t-1}))`, `k_t = W_k · (μ_k x_t + (1-μ_k) x_{t-1})`.
- `y_t = r_t ⊙ σ²(k_t)` — squared-key (squared ReLU-like) nonlinearity gated by receptance.

## Used By

| Model | Role |
|-------|------|
| RWKV | FFN sub-block alternating with time-mixing in every layer |

## Features

- **Gated FFN with built-in time context** — no separate position handling.
- **Fully parallel training** — same RNN-as-transformer trick as time-mixing.

## Evolution

- **Predecessor**: swiglu-ffn (gated FFN without time context).
- **Successor**: RWKV-4/5/6 replace channel-mixing internals (Goose/Eagle kernels) while keeping the time/channel split.
