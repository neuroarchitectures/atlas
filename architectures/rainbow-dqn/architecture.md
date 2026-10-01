# Architecture: Rainbow DQN

## Motivation

DQN achieved human-level Atari performance but had several limitations: overestimation bias, uniform sampling, scalar Q-values. Rainbow combines six independent improvements into one agent.

## Core Idea

Combine: (1) Double DQN (reduce overestimation), (2) Prioritized Experience Replay, (3) Dueling architecture (separate V and A), (4) Multi-step returns (n-step TD), (5) Distributional Q (C51), (6) Noisy Nets (exploration via parameter noise).

## Architecture

### Overview

![rainbow dqn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (10 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | State (Frame) | `input` | shape: [4, 84, 84] |
| 2 | Conv 1 | `conv` | outChannels: 32, kernelSize: 8, stride: 4 |
| 3 | ReLU | `relu` |  |
| 4 | Conv 2 | `conv` | outChannels: 64, kernelSize: 4, stride: 2 |
| 5 | ReLU 2 | `relu` |  |
| 6 | Conv 3 | `conv` | outChannels: 64, kernelSize: 3, stride: 1 |
| 7 | Flatten | `flatten` |  |
| 8 | Value Stream | `linear` | outFeatures: 51 (C51 distributional) |
| 8 | Advantage Stream | `linear` | outFeatures: 51 * n_actions |
| 9 | Q Distribution | `output` | V + A - mean(A) |

</details>

### Components

1. **Double DQN** — Use the online network for action selection and target network for evaluation, reducing overestimation. 2. **Prioritized Experience Replay** — Sample transitions proportional to TD error, with importance sampling correction. 3. **Dueling architecture** — Separate V(s) and A(s,a) streams: Q = V + A - mean(A). 4. **Multi-step returns** — Use n-step TD targets for faster credit assignment. 5. **Distributional (C51)** — Predict Q-value distribution over 51 atoms, not scalar. 6. **Noisy Nets** — Replace epsilon-greedy with parameter-space noise for state-dependent exploration.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Combine: (1) Double DQN (reduce overestimation), (2) Prioritized Experience Replay, (3) Dueling architecture (separate V and A), (4) Multi-step returns (n-step TD), (5) Distributional Q (C51), (6) Noi
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DQN, Double DQN, Dueling DQN. Successor: IQN, FQF, RAINBOW variants.

## References

- Hessel et al. 2018
