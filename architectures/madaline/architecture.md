# Architecture: MADALINE

## Motivation

A single ADALINE could only solve linearly separable problems. To address XOR-like problems, multiple ADALINE units needed to be combined in layers.

## Core Idea

MADALINE (Multiple ADALINE) consists of multiple ADALINE units arranged in layers with a majority-vote output rule. The training algorithm uses a minimal disturbance principle — only the ADALINE unit whose output is closest to the decision boundary is adjusted.

## Architecture

### Overview

![madaline architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [2] |
| 2 | ADALINE-1 | `linear` | outFeatures: 1, inFeatures: 2 |
| 2 | ADALINE-2 | `linear` | outFeatures: 1, inFeatures: 2 |
| 2 | ADALINE-3 | `linear` | outFeatures: 1, inFeatures: 2 |
| 3 | Majority Vote | `vote` |  |
| 4 | Output | `output` |  |

</details>

### Components

1. **Multiple ADALINE units** — Each unit is a linear neuron with identity activation and sign quantizer. 2. **Majority vote** — The output layer uses a hard majority voting scheme among the ADALINE units. 3. **Minimal disturbance training** — Only the unit closest to the boundary is updated, minimizing disruption to other units.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — MADALINE (Multiple ADALINE) consists of multiple ADALINE units arranged in layers with a majority-vote output rule. The training algorithm uses a minimal disturbance principle — only the ADALINE unit 
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ADALINE (1960). Successor: Backpropagation-trained multi-layer networks (1986).

## References

- Widrow 1962
