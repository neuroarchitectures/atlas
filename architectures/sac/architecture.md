# Architecture: Soft Actor-Critic

## Motivation

On-policy RL methods (PPO) are sample-inefficient. SAC uses off-policy learning with a maximum entropy objective that encourages exploration, achieving both high performance and sample efficiency.

## Core Idea

Maximize expected reward plus entropy: J = E[sum(r + alpha * H(pi))]. Twin Q-networks (clipped double Q) reduce overestimation. Stochastic policy with reparameterization enables backpropagation through the policy.

## Architecture

### Overview

![soft actor-critic architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (10 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | State | `input` | shape: [17] |
| 2 | Policy Encoder | `linear` | outFeatures: 256 |
| 3 | ReLU | `relu` |  |
| 4 | Mean | `linear` | outFeatures: 6 |
| 4 | LogStd | `linear` | outFeatures: 6 |
| 5 | Reparameterize | `identity` | a = mu + std * epsilon |
| 6 | Q1 Network | `linear` | twin Q, outFeatures: 1 |
| 6 | Q2 Network | `linear` | twin Q, outFeatures: 1 |
| 7 | Action | `output` |  |
| 7 | Q Value | `output` | min(Q1, Q2) |

</details>

### Components

1. **Stochastic policy** — Outputs Gaussian (mean, log_std) and samples via reparameterization trick. 2. **Twin Q-networks** — Two independent Q-networks; the minimum is used as the target, reducing overestimation bias. 3. **Maximum entropy objective** — J = E[r + alpha * H(pi)], where alpha is automatically tuned to maintain target entropy. 4. **Off-policy** — Uses replay buffer for sample-efficient learning. 5. **Target networks** — Exponential moving average target Q-networks for stability.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Maximize expected reward plus entropy: J = E[sum(r + alpha * H(pi))]. Twin Q-networks (clipped double Q) reduce overestimation. Stochastic policy with reparameterization enables backpropagation throug
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DDPG, TD3. Successor: SAC variants (SAC-AE, SAC+AE).

## References

- Haarnoja et al. 2018
