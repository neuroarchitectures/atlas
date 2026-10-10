# Architecture: Dueling DQN

## Motivation

DQN with separate state value and advantage streams for better policy evaluation.

## Core Idea

DQN with separate state value and advantage streams for better policy evaluation.

## Architecture

### Overview

![dueling-dqn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Game Frame | `input` | shape=[4, 84, 84] |
| 2 | Conv 1 | `conv2d` | outFeatures=32 |
| 3 | Conv 2 | `conv2d` | outFeatures=64 |
| 4 | Conv 3 | `conv2d` | outFeatures=512 |
| 5 | FC Layer | `linear` | outFeatures=512 |
| 6 | Value Stream V(s) | `linear` | outFeatures=1 |
| 7 | Advantage Stream A(s,a) | `linear` | outFeatures=18 |
| 8 | Q-values | `output` | — |

</details>

### Components

1. **Game Frame** — input layer. 2. **Conv 1** — conv2d layer. 2. **Conv 2** — conv2d layer. 2. **Conv 3** — conv2d layer. 2. **FC Layer** — linear layer. 2. **Value Stream V(s)** — linear layer. 2. **Advantage Stream A(s,a)** — linear layer. 2. **Q-values** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — DQN with separate state value and advantage streams for better policy evaluation.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
