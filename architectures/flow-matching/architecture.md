# Architecture: Flow Matching

## Motivation

Score-based diffusion models require solving SDEs which are computationally expensive. Flow matching simplifies this by training an ODE-based flow with a simpler, more direct training objective.

## Core Idea

Define a target vector field u_t(x) that transports samples from a simple prior (Gaussian) to the data distribution along straight-line (optimal transport) paths. Train a network v_theta(x, t) to match this target field via regression.

## Architecture

### Overview

![flow matching architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Sample x_t | `input` | shape: [3, 32, 32] |
| 2 | Time Embedding | `embedding` | numEmbeddings: 1000, embeddingDim: 128 |
| 3 | Vector Field v(x,t) | `conv` | estimates velocity field for ODE |
| 4 | Velocity Field | `output` | used in ODE: dx/dt = v_theta(x, t) |

</details>

### Components

1. **Vector field network** — A UNet v_theta(x, t) that estimates the velocity field of the flow. 2. **Optimal transport paths** — The target vector field u_t is defined along straight-line (OT) paths from noise to data: x_t = (1-t)*x_1 + t*x_0. 3. **Flow matching loss** — ||v_theta(x_t, t) - u_t(x_t|x_0, x_1)||^2, a simple regression. 4. **ODE sampling** — Generate by integrating dx = v_theta(x, t) dt from t=0 to t=1.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Define a target vector field u_t(x) that transports samples from a simple prior (Gaussian) to the data distribution along straight-line (optimal transport) paths. Train a network v_theta(x, t) to matc
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Score-based diffusion. Successor: Rectified Flow, Stable Diffusion 3.

## References

- Lipman et al. 2023
