# Architecture: AlphaZero

## Motivation

Self-play reinforcement learning with MCTS + deep residual network for board game mastery.

## Core Idea

Self-play reinforcement learning with MCTS + deep residual network for board game mastery.

## Architecture

### Overview

![alphazero architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Board State | `input` | shape=[17, 8, 8] |
| 2 | ResNet Backbone (19 blocks) | `conv2d` | outFeatures=256 |
| 3 | Policy Head | `conv2d` | outFeatures=73 |
| 4 | Value Head | `linear` | outFeatures=1 |
| 5 | Policy + Value | `output` | — |

</details>

### Components

1. **Board State** — input layer. 2. **ResNet Backbone (19 blocks)** — conv2d layer. 2. **Policy Head** — conv2d layer. 2. **Value Head** — linear layer. 2. **Policy + Value** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Self-play reinforcement learning with MCTS + deep residual network for board game mastery.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
