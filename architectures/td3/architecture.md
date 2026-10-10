# Architecture: TD3

## Motivation

DDPG suffers from overestimation bias in the Q-function, leading to suboptimal policies in continuous action spaces.

## Core Idea

Use twin Q-networks (take the min to reduce overestimation), delay policy updates relative to Q updates, and add target policy smoothing.

## Architecture

### Overview

![td3 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | State | `input` | shape: [17] |
| 2 | Actor Hidden | `linear` | outFeatures: 256 |
| 3 | ReLU | `relu` |  |
| 4 | Actor Output | `linear` | outFeatures: 6 |
| 5 | Tanh | `tanh` |  |
| 6 | Action | `output` |  |

</details>

### Components

1. **Actor network** — Deterministic policy mapping states to continuous actions. 2. **Twin critics** — Two independent Q-networks; use min(Q1, Q2) for target. 3. **Delayed updates** — Update policy less frequently than critics. 4. **Target smoothing** — Add clipped noise to target actions.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use twin Q-networks (take the min to reduce overestimation), delay policy updates relative to Q updates, and add target 
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DDPG. Successor: SAC.

## References

Fujimoto et al. 2018
