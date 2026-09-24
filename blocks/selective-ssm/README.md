# Selective State Space Model (Mamba Block)

## Design Philosophy

Make the SSM parameters (Δ, B, C) **input-dependent** so the model can selectively propagate or forget information depending on the current token. The philosophy: prior SSMs (S4) are linear time-invariant (LTI) — their dynamics are constant across time steps, so they can't do content-based selection. Mamba breaks LTI to gain selectivity, accepting the loss of convolutional computation and replacing it with a hardware-aware parallel scan. The block combines the SSM and the MLP into a single homogeneous gated block.

## Functionality

1. **Input projection**: `x → Linear` produces input-dependent `B_t, C_t, Δ_t` (via `softplus` for Δ).
2. **Discretization**: `(Δ_t, A, B_t) → (Ā_t, B̄_t)` via zero-order hold: `Ā = exp(ΔA)`, `B̄ = (ΔA)^{-1}(exp(ΔA) - I)·ΔB`.
3. **State update (recurrent)**: `h_t = Ā_t · h_{t-1} + B̄_t · x_t` — input-dependent dynamics.
4. **Output**: `y_t = C_t · h_t`.
5. **Gating**: SSM output × a gated branch (multiplicative control).
6. **Residual**: Gated output + input.
- A is diagonal (inherited from S4).

## Used By

| Model | Role |
|-------|------|
| Mamba | The defining block |
| Mamba-2 / Mamba-3 | Improved hardware utilization |
| Jamba | Hybrid Mamba-Transformer blocks |

## Features

- **Linear-time inference**: O(1) per step, no KV cache.
- **Content-based selection**: Input-dependent Δ/B/C enable forgetting irrelevant tokens.
- **Hardware-aware**: Parallel scan keeps computation in GPU SRAM.
- **No attention, no MLP**: A single homogeneous block.

## Evolution

- **Predecessor**: S4 (LTI SSM, computed as convolutions); H3; Hyena.
- **Successor**: Mamba-2/3; Jamba (hybrid with attention); various Mamba-Transformer hybrids.
