# Architecture: A2C

## Motivation

Advantage Actor-Critic: shared CNN encoder with separate policy and value heads, synchronous updates.

## Core Idea

Advantage Actor-Critic: shared CNN encoder with separate policy and value heads, synchronous updates.

## Architecture

### Overview

![a2c architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Game Frame | `input` | shape=[4, 84, 84] |
| 2 | Conv 1 | `conv2d` | outFeatures=32 |
| 3 | Conv 2 | `conv2d` | outFeatures=64 |
| 4 | Shared FC | `linear` | outFeatures=512 |
| 5 | Policy Head (actor) | `linear` | outFeatures=6 |
| 6 | Value Head (critic) | `linear` | outFeatures=1 |
| 7 | Policy + Value | `output` | — |

</details>

### Components

1. **Game Frame** — input layer. 2. **Conv 1** — conv2d layer. 2. **Conv 2** — conv2d layer. 2. **Shared FC** — linear layer. 2. **Policy Head (actor)** — linear layer. 2. **Value Head (critic)** — linear layer. 2. **Policy + Value** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Advantage Actor-Critic: shared CNN encoder with separate policy and value heads, synchronous updates.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
