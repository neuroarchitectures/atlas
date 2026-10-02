# Architecture: REINFORCE

## Motivation

Value-based methods (DQN) struggle with continuous or large action spaces. Direct policy optimization is needed.

## Core Idea

Directly parameterize the policy pi(a|s) and optimize the expected return J = E[log pi(a|s) * G] using the policy gradient theorem, where G is the return.

## Architecture

### Overview

![reinforce architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | State | `input` | shape: [4] |
| 2 | Hidden | `linear` | outFeatures: 128 |
| 3 | ReLU | `relu` |  |
| 4 | Policy Head | `linear` | outFeatures: 2 |
| 5 | Softmax | `softmax` |  |
| 6 | Action Probabilities | `output` |  |

</details>

### Components

1. **Policy network** — Neural network parameterizing pi(a|s). 2. **Stochastic sampling** — Actions sampled from the policy distribution. 3. **Return weighting** — Log-probability of actions weighted by episode return. 4. **No replay buffer** — On-policy learning.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Directly parameterize the policy pi(a|s) and optimize the expected return J = E[log pi(a|s) * G] using the policy gradie
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: (foundational). Successor: A2C, PPO, TRPO.

## References

Williams 1992
