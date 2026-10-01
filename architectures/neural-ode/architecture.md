# Architecture: Neural ODE

## Motivation

Deep networks with many layers can be viewed as Euler discretizations of a continuous ODE. Neural ODEs parametrize the derivative of the hidden state directly, enabling arbitrary-depth evaluation via ODE solvers and memory-efficient training via the adjoint method.

## Core Idea

Define dh/dt = f(h(t), t, theta) where f is a neural network. The output is obtained by integrating this ODE from t=0 to t=T using a black-box ODE solver (e.g., RK4). Training uses the adjoint sensitivity method for O(1) memory.

## Architecture

### Overview

![neural ode architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [64] — initial state h(0) |
| 2 | ODE Function f(h,t) | `linear` | parametrizes dh/dt = f(h(t), t, theta) |
| 3 | ODE Solver (RK4) | `identity` | integrates from t=0 to t=T |
| 4 | Output | `output` | h(T) — final state |

</details>

### Components

1. **ODE function f(h, t, theta)** — A neural network (typically an MLP) that defines the derivative of the hidden state with respect to continuous 'depth' t. 2. **ODE solver** — A black-box ODE solver (e.g., Dormand-Prince/RK45) integrates the ODE to obtain h(T) from h(0). The solver adaptively chooses step sizes. 3. **Adjoint method** — Backpropagation uses the adjoint sensitivity method, which solves a reverse-time ODE and requires O(1) memory regardless of the effective depth.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Define dh/dt = f(h(t), t, theta) where f is a neural network. The output is obtained by integrating this ODE from t=0 to t=T using a black-box ODE solver (e.g., RK4). Training uses the adjoint sensiti
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ResNet (as continuous-depth limit). Successor: Neural SDEs, Neural CDEs, Continuous Normalizing Flows.

## References

- Chen et al. 2018
