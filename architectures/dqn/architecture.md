# Architecture: Deep Q-Network

## Motivation

Tabular Q-learning fails for high-dimensional state spaces (e.g., raw pixels). DQN uses a deep CNN to approximate Q(s,a), enabling RL directly from pixel input.

## Core Idea

Stack 4 frames as state. Three convolutional layers extract spatial features. A fully connected layer outputs Q-values for all actions. Experience replay + target network stabilize training.

## Architecture

### Overview

![dqn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Game Frame | `input` | 4x84x84 |
| 2 | Conv Layer 1 | `conv2d` | 32 filters, 8x8, stride 4 |
| 3 | Conv Layer 2 | `conv2d` | 64 filters, 4x4, stride 2 |
| 4 | Conv Layer 3 | `conv2d` | 64 filters, 3x3, stride 1 |
| 5 | Q-Values | `linear` | 512 -> num_actions |
| 6 | Action Values | `output` | Q(s,a) per action |

</details>

### Components

1. **Frame stacking** — 4 consecutive frames capture motion. 2. **CNN feature extractor** — 3 conv layers process game pixels. 3. **Q-value head** — outputs Q(s,a) for all actions simultaneously. 4. **Experience replay** — store transitions in buffer, sample mini-batches. 5. **Target network** — separate network with delayed updates for stable targets. 6. **Epsilon-greedy** — balance exploration and exploitation.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Stack 4 frames as state. Three convolutional layers extract spatial features. A fully connected layer outputs Q-values f...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Q-learning (tabular). Successor: Double DQN, Dueling DQN, Rainbow DQN.

## References

- Mnih et al. 2015 (Nature)
