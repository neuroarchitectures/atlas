# Structured State Space Model (S4)

## Design Philosophy

A state space model `x'(t) = Ax(t) + Bu(t)`, `y(t) = Cx(t)` with a **structured A matrix** (HiPPO) that enables efficient long-range modeling. The philosophy: the SSM unifies continuous-time, RNN, and CNN views — train as a convolution (parallel, via FFT), infer as a recurrence (constant memory). The HiPPO matrix lets the state memorize input history, addressing 10000+ step dependencies.

## Functionality

- **SSM**: `x'(t) = Ax(t) + Bu(t)`, `y(t) = Cx(t) + Du(t)`. D is a skip connection.
- **HiPPO matrix A**: A special matrix (from HiPPO theory) that lets `x(t)` memorize the history of `u(t)`.
- **Bilinear discretization**: Continuous → discrete: `Ā = (I - Δ/2·A)^{-1}(I + Δ/2·A)`.
- **Convolution kernel K**: `y = K * u`, where `K = (CB, CAB, ..., CA^{L-1}B)`. Computed via FFT.
- **NPLR/DPLR parameterization**: `A = VΛV* - PQ*` (Normal Plus Low-Rank) → diagonalizable + Woodbury correction → Cauchy kernel computation in `Õ(N + L)`.

## Used By

| Model | Role |
|-------|------|
| S4 | The defining model |
| S4D (diagonal S4) | Simplified diagonal A |
| H3 | SSM architecture with gated connections |
| Mamba (foundation) | Mamba builds on S4's SSM + diagonal A |

## Features

- **Dual view**: Convolution (parallel training) / recurrence (constant-memory inference).
- **Long-range**: HiPPO A enables 10000+ step dependencies.
- **Õ(N + L) compute**: Near-linear in state dim and sequence length.

## Evolution

- **Predecessor**: LSSL (Linear State Space Layer, infeasible); HiPPO theory.
- **Successor**: S4D (diagonal, simpler); Mamba (input-dependent, breaks LTI for selectivity).
