# Architecture: MuZero

## Motivation

AlphaZero needs to know the environment's rules (forward model) for planning. MuZero learns an implicit model of the environment, enabling planning without domain knowledge.

## Core Idea

Three networks: representation (observation to latent state), dynamics (latent state + action to next latent state + reward), and prediction (latent state to policy + value). MCTS searches in latent space using the learned model.

## Architecture

### Overview

![muzero architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Observation | `input` | raw pixels |
| 2 | Representation | `linear` | obs -> latent state |
| 3 | Dynamics | `linear` | state + action -> next state + reward |
| 4 | Prediction | `linear` | state -> policy + value |
| 5 | Policy + Value | `output` | action probs + V(s) |

</details>

### Components

1. **Representation function h(o)** — encodes observation into latent state. 2. **Dynamics function g(s,a)** — predicts next latent state and reward. 3. **Prediction function f(s)** — outputs policy and value from latent state. 4. **MCTS in latent space** — Monte Carlo tree search using learned model. 5. **Reanalysis** — re-evaluate past trajectories with updated model. 6. **No environment model needed** — learns the dynamics from data.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Three networks: representation (observation to latent state), dynamics (latent state + action to next latent state + rew...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: AlphaZero (needs rules). Successor: EfficientZero, Stochastic MuZero.

## References

- Schrittwieser et al. 2020
