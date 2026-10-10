# Architecture: World Model

## Motivation

RL agents that learn directly from environment interaction are sample-inefficient. A world model that can predict environment dynamics enables learning entirely in imagination.

## Core Idea

Encode observations to a latent state, predict future latent states and rewards using a recurrent model (RSSM), and decode observations. The agent learns to act by rolling out the world model.

## Architecture

### Overview

![world model architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Observation | `input` | shape: [3, 64, 64] |
| 2 | Encoder | `conv` | CNN encoder to latent |
| 3 | RSSM | `lstm` | Recurrent State-Space Model |
| 4 | Dynamics Predictor | `linear` | predict next latent state |
| 4 | Decoder | `convTranspose` | reconstruct observation |
| 4 | Reward Predictor | `linear` | predict reward |
| 5 | Next State | `output` |  |
| 5 | Reconstruction | `output` |  |
| 5 | Reward | `output` |  |

</details>

### Components

1. **RSSM (Recurrent State-Space Model)** — A variational model that maintains a stochastic latent state with both deterministic (LSTM) and stochastic components. 2. **Encoder** — Maps observations to latent states (CNN for images). 3. **Decoder** — Reconstructs observations from latent states (for training signal). 4. **Reward predictor** — Predicts reward from latent state. 5. **Actor-critic** — Learns to act by rolling out the world model in imagination, then distilling to the actor.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Encode observations to a latent state, predict future latent states and rewards using a recurrent model (RSSM), and decode observations. The agent learns to act by rolling out the world model.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Dreamer. Successor: DreamerV2/V3, GAIA-1, Genie.

## References

- Ha & Schmidhuber 2018; Hafner et al. 2019
