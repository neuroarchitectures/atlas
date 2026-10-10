# Architecture: TRPO

## Motivation

Trust Region Policy Optimization: constrained policy gradient with KL divergence trust region.

## Core Idea

Trust Region Policy Optimization: constrained policy gradient with KL divergence trust region.

## Architecture

### Overview

![trpo architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | State | `input` | shape=[17] |
| 2 | Shared FC 1 | `linear` | outFeatures=64 |
| 3 | Shared FC 2 | `linear` | outFeatures=64 |
| 4 | Policy Head | `linear` | outFeatures=6 |
| 5 | Value Head | `linear` | outFeatures=1 |
| 6 | Policy + Value | `output` | — |

</details>

### Components

1. **State** — input layer. 2. **Shared FC 1** — linear layer. 2. **Shared FC 2** — linear layer. 2. **Policy Head** — linear layer. 2. **Value Head** — linear layer. 2. **Policy + Value** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Trust Region Policy Optimization: constrained policy gradient with KL divergence trust region.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
