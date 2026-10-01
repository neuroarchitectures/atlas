# Architecture: Echo State Network

## Motivation

Training RNNs with backpropagation through time is computationally expensive and unstable. ESNs exploit the observation that a large random dynamical system can serve as a universal approximator if only a linear readout is trained.

## Core Idea

Maintain a large fixed random recurrent reservoir with the 'echo state property' (spectral radius < 1). Only train a linear readout layer mapping reservoir states to outputs, typically via ridge regression.

## Architecture

### Overview

![echo state network architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [1] |
| 2 | Input Weights (Fixed) | `linear` | outFeatures: 500, inFeatures: 1 |
| 3 | Reservoir (Fixed) | `lstm` | hiddenSize: 500, spectral radius < 1 |
| 4 | Readout (Trained) | `linear` | outFeatures: 1, inFeatures: 500 |
| 5 | Output | `output` |  |

</details>

### Components

1. **Input weights** — Fixed random projection from input to reservoir. 2. **Reservoir** — A large fixed random recurrent network with spectral radius < 1 (echo state property). The reservoir acts as a nonlinear dynamical feature map. 3. **Readout** — The only trained component, typically a linear layer trained via ridge regression or least squares.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Maintain a large fixed random recurrent reservoir with the 'echo state property' (spectral radius < 1). Only train a linear readout layer mapping reservoir states to outputs, typically via ridge regre
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: RNN. Related: Liquid State Machine (Maass 2002). Successor: Reservoir computing variants.

## References

- Jaeger 2001
