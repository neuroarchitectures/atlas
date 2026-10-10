# Architecture: Rectified Flow

## Motivation

Flow matching paths can be curved, requiring many ODE steps. Rectified flow 'rectifies' the paths to be straighter, enabling faster sampling with fewer steps.

## Core Idea

Start with a flow matching model, then retrain on the trajectories of the learned flow. This 'rectification' step straightens the transport paths, reducing the curvature and enabling fewer-step generation.

## Architecture

### Overview

![rectified flow architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | x_t | `input` | shape: [3, 32, 32] |
| 2 | Time Embedding | `embedding` | numEmbeddings: 1000, embeddingDim: 128 |
| 3 | Rectified Vector Field | `conv` | straight-line OT transport |
| 4 | Velocity | `output` | few-step ODE sampling |

</details>

### Components

1. **Rectified vector field** — After initial flow matching, the model is retrained on trajectories from the first model, making paths straighter. 2. **Straight-line transport** — The rectified flow approximates x_t = (1-t)*x_0 + t*x_1, with velocity dx/dt = x_1 - x_0 (constant). 3. **Few-step sampling** — Straighter paths allow Euler ODE with 1-10 steps instead of 100-1000.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Start with a flow matching model, then retrain on the trajectories of the learned flow. This 'rectification' step straightens the transport paths, reducing the curvature and enabling fewer-step genera
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Flow Matching. Related: Stable Diffusion 3 (uses rectified flow).

## References

- Liu et al. 2023
