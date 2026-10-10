# Architecture: A3C

## Motivation

Experience replay in DQN requires large memory and is sample-inefficient. Single-thread training is slow.

## Core Idea

Use asynchronous gradient descent with multiple parallel worker threads exploring different parts of the environment simultaneously, stabilizing training without replay buffers.

## Architecture

### Overview

![a3c architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | State | `input` | shape: [4] |
| 2 | Conv2d | `conv2d` |  |
| 3 | Conv2d | `conv2d` |  |
| 4 | Flatten | `flatten` |  |
| 5 | Shared FC | `linear` | outFeatures: 256 |
| 6 | ReLU | `relu` |  |
| 7 | Actor | `linear` | outFeatures: 4 |
| 8 | Critic | `linear` | outFeatures: 1 |
| 9 | Action / Value | `output` |  |

</details>

### Components

1. **Shared CNN encoder** — Conv layers extract spatial features from game frames. 2. **Actor head** — Outputs policy distribution over actions. 3. **Critic head** — Estimates state value V(s). 4. **Async updates** — Multiple threads update a shared model asynchronously.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use asynchronous gradient descent with multiple parallel worker threads exploring different parts of the environment sim
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DQN. Successor: A2C, PPO, IMPALA.

## References

Mnih et al. 2016
