# Complex MIMO State-Space Mixer (Mamba-3)

## Design Philosophy

Push the selective SSM further on three axes: *complex-valued* state (equivalent to data-dependent RoPE, enabling state tracking that real-valued SSMs provably can't), *trapezoidal* discretization (a superset of ZOH), and a *matrix-multiply MIMO* state update that raises decode-time arithmetic intensity.

## Functionality

- Complex state update: `s_t = Ā(Δ,x) ⊙ s_{t-1} + B̄(Δ,x) x_t` with Ā, B̄ complex; trapezoidal rule discretization.
- MIMO: multiply a matrix of states by the input matrix per step (better tensor-core utilization at decode); QK-norm replaces the pre-output norm.

## Used By

| Model | Role |
|-------|------|
| Mamba-3 | Gated SSM block + FFN residual pairs, selective (B, C, Δ) |

## Features

- **State tracking capability** — complex dynamics recover expressive power real SSMs lack.
- **Decode throughput** — MIMO form keeps the kernel efficient at batch-1.

## Evolution

- **Predecessor**: selective-ssm (Mamba/Mamba-2 real-valued scalar-per-state update).
- **Related**: gated DeltaNet / GLA — neighboring linear-attention-family mixers (as in Nemotron's hybrid stack).
