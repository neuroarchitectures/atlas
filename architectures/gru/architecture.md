# Architecture: GRU

## Motivation

LSTM has 3 gates (input, forget, output) and a separate cell state, making it parameter-heavy. GRU simplifies this to 2 gates (update, reset) and merges the cell and hidden states, reducing parameters while maintaining similar performance on most tasks.

## Core Idea

The GRU has two gates: (1) Update gate z_t controls how much of the previous state to keep, (2) Reset gate r_t controls how much of the previous state to use when computing the candidate. The new state is: h_t = (1-z_t)·h_{t-1} + z_t·h̃_t, where h̃_t is the candidate. This merges the forget and input gates into a single update gate.

## Architecture

### Overview

![gru architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (12 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input x_t | `input` | shape: [1, 100] |
| 2 | Update Gate z_t | `linear` | outFeatures: 256, inFeatures: 356 |
| 3 | Reset Gate r_t | `linear` | outFeatures: 256, inFeatures: 356 |
| 4 | Candidate ĥ_t | `linear` | outFeatures: 256, inFeatures: 356 |
| 5 | σ | `sigmoid` |  |
| 6 | σ | `sigmoid` |  |
| 7 | Reset: r_t ⊙ h_{t-1} | `custom` | type: elementwise_multiply |
| 8 | tanh | `tanh` |  |
| 9 | (1-z_t) ⊙ h_{t-1} | `custom` | type: complement_multiply |
| 10 | z_t ⊙ ĥ_t | `custom` | type: elementwise_multiply |
| 11 | h_t | `add` |  |
| 12 | Output h_t | `output` |  |

</details>

The GRU has two gates: (1) Update gate z_t controls how much of the previous state to keep, (2) Reset gate r_t controls how much of the previous state to use when computing the candidate. The new state is: h_t = (1-z_t)·h_{t-1} + z_t·h̃_t, where h̃_t is the candidate. This merges the forget and input gates into a single update gate.

### Components

2. **Update Gate z_t** (`linear`, scope: `gru`) — Params: outFeatures: 256, inFeatures: 356
3. **Reset Gate r_t** (`linear`, scope: `gru`) — Params: outFeatures: 256, inFeatures: 356
4. **Candidate ĥ_t** (`linear`, scope: `gru`) — Params: outFeatures: 256, inFeatures: 356
5. **σ** (`sigmoid`, scope: `gru`) — Params: none
6. **σ** (`sigmoid`, scope: `gru`) — Params: none
7. **Reset: r_t ⊙ h_{t-1}** (`custom`, scope: `gru`) — Params: type: elementwise_multiply
8. **tanh** (`tanh`, scope: `gru`) — Params: none
9. **(1-z_t) ⊙ h_{t-1}** (`custom`, scope: `gru`) — Params: type: complement_multiply
10. **z_t ⊙ ĥ_t** (`custom`, scope: `gru`) — Params: type: elementwise_multiply
11. **h_t** (`add`, scope: `gru`) — Params: none

### Data Flow

The architecture processes input through a sequence of 12 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to GRU.

## Evolution

GRU is a simplification of LSTM with fewer parameters and comparable performance. It is widely used in NLP, speech, and time-series tasks. Successors: SRU (simple recurrent unit), IndRNN, and modern SSMs (S4, Mamba) which further simplify recurrent computation while enabling parallel training.

## Source

- **Paper:** arXiv:1406.1078
- **Year:** 2014
- **Authors:** Cho et al.
