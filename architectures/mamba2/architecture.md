# Architecture: Mamba-2

## Motivation

Mamba's selective SSM is powerful but its recurrent formulation limits parallelism. Mamba-2 connects SSMs to attention, enabling parallel scan and structured state space (SSD) formulation for faster training.

## Core Idea

Reformulates the SSM as a structured attention-like computation (State Space Duality). This enables parallel scan training (faster than Mamba's recurrent scan) while maintaining linear inference cost. Larger state dimension (d_state) improves expressiveness.

## Architecture

### Overview

![mamba2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | token ids |
| 2 | Token Embedding | `linear` | 128K vocab -> 4096 |
| 3 | SSM Layers | `linear` | 64 layers, SSD |
| 4 | RMSNorm | `identity` | final norm |
| 5 | LM Head | `linear` | 4096 -> 128K |
| 6 | Next Token | `output` | autoregressive |

</details>

### Components

1. **State Space Duality (SSD)** — SSM = structured attention. 2. **Parallel scan** — faster training than recurrent formulation. 3. **Larger state** — d_state=128 (vs 16 in Mamba). 4. **Selective mechanism** — input-dependent state transitions. 5. **Multi-head SSM** — grouped SSM heads like multi-head attention. 6. **Linear inference** — O(N) at inference time.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Reformulates the SSM as a structured attention-like computation (State Space Duality). This enables parallel scan traini...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Mamba (selective SSM), S6. Successor: Jamba (MoE+SSM), Zamba.

## References

- Dao & Gu 2024
