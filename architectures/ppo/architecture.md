# Architecture: PPO

## Motivation

Policy gradient methods are sensitive to step size. Too large a step can destroy the policy. PPO constrains policy updates to a trust region without requiring second-order optimization.

## Core Idea

Use a clipped surrogate objective that prevents the new policy from deviating too far from the old policy. The ratio r = pi_new(a|s) / pi_old(a|s) is clipped to [1-eps, 1+eps].

## Architecture

### Overview

![ppo architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | State | `input` | shape: [17] |
| 2 | Shared Encoder | `linear` | outFeatures: 64 |
| 3 | Tanh | `tanh` |  |
| 4 | Actor (Policy) | `linear` | outFeatures: 6 (action mean) |
| 4 | Critic (Value) | `linear` | outFeatures: 1 (state value) |
| 5 | Action (Mean) | `output` | Gaussian policy |
| 5 | State Value | `output` | baseline for advantage |

</details>

### Components

1. **Actor (policy network)** — Outputs action distribution parameters (mean and std for continuous, logits for discrete). 2. **Critic (value network)** — Estimates V(s), the expected return from state s. 3. **Clipped surrogate objective** — L = min(r * A, clip(r, 1-eps, 1+eps) * A), where r is the probability ratio and A is the advantage. 4. **Generalized Advantage Estimation (GAE)** — Computes advantage estimates using temporal difference residuals. 5. **Multiple epochs** — Unlike vanilla policy gradient, PPO performs multiple epochs of SGD on the same data.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use a clipped surrogate objective that prevents the new policy from deviating too far from the old policy. The ratio r = pi_new(a|s) / pi_old(a|s) is clipped to [1-eps, 1+eps].
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: TRPO. Successor: PPG, DAAC, PPO variants.

## References

- Schulman et al. 2017
