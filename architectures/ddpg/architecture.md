# Architecture: DDPG

## Motivation

DQN handles discrete actions but many RL problems require continuous action spaces (robotics, control).

## Core Idea

Combine the deterministic policy gradient theorem with DQN's innovations (replay buffer, target networks) to learn deterministic policies in continuous action spaces.

## Architecture

### Overview

![ddpg architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | State | `input` | shape: [17] |
| 2 | Actor Hidden | `linear` | outFeatures: 400 |
| 3 | ReLU | `relu` |  |
| 4 | Actor Hidden 2 | `linear` | outFeatures: 300 |
| 5 | ReLU | `relu` |  |
| 6 | Action Output | `linear` | outFeatures: 6 |
| 7 | Tanh | `tanh` |  |
| 8 | Continuous Action | `output` |  |

</details>

### Components

1. **Actor** — Deterministic policy mu(s) mapping states to actions. 2. **Critic** — Q(s, a) estimating action-value. 3. **Replay buffer** — Off-policy learning from stored transitions. 4. **Target networks** — Soft-updated copies for stability. 5. **Exploration** — Ornstein-Uhlenbeck noise added to actions.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Combine the deterministic policy gradient theorem with DQN's innovations (replay buffer, target networks) to learn deter
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DQN, deterministic policy gradient. Successor: TD3, SAC.

## References

Lillicrap et al. 2016
