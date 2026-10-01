# Architecture: Decision Transformer

## Motivation

Traditional RL requires online interaction and value estimation. Decision Transformer reframes RL as sequence modeling, enabling offline learning from trajectories by treating (return, state, action) as tokens.

## Core Idea

Feed (return-to-go, state, action) triplets through a GPT-style transformer. At inference, condition on a desired return and let the transformer generate the optimal action sequence.

## Architecture

### Overview

![decision transformer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Return-to-Go | `input` | shape: [1] |
| 1 | State | `input` | shape: [17] |
| 1 | Action | `input` | shape: [6] |
| 2 | RTG Embed | `linear` | outFeatures: 128 |
| 2 | State Embed | `linear` | outFeatures: 128 |
| 2 | Action Embed | `linear` | outFeatures: 128 |
| 3 | Transformer | `linear` | GPT-style, causal attention |
| 4 | Action | `output` | predicted action |

</details>

### Components

1. **Return-to-go (RTG)** — The target cumulative return from the current timestep. Conditions the transformer on desired performance. 2. **Sequence of triplets** — Input is (R_1, s_1, a_1, R_2, s_2, a_2, ...). Causal attention ensures each token only sees past tokens. 3. **GPT architecture** — Standard transformer decoder with layer norm and causal self-attention. 4. **Inference** — Set RTG to desired return, feed current state, predict action. Roll forward.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Feed (return-to-go, state, action) triplets through a GPT-style transformer. At inference, condition on a desired return and let the transformer generate the optimal action sequence.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: GPT, Trajectory Transformer. Successor: Online Decision Transformer, Q-learning + DT hybrids.

## References

- Chen et al. 2021
