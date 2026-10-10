# Architecture: IMPALA

## Motivation

Scaling RL to many environments requires decoupling acting from learning. IMPALA uses distributed actors and a central learner, with V-trace to correct for the off-policy data.

## Core Idea

Many actor processes run the policy in parallel, collecting trajectories. A central learner trains on batches of trajectories using V-trace, a truncated importance sampling correction. The CNN+LSTM policy processes visual observations.

## Architecture

### Overview

![impala architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Observation | `input` | 3x84x84 |
| 2 | CNN Encoder | `conv2d` | deep ResNet |
| 3 | LSTM | `lstm` | hidden=256 |
| 4 | Policy Head | `linear` | num_actions |
| 5 | Value Head | `linear` | 1 |
| 6 | Action + Value | `output` | policy + V(s) |

</details>

### Components

1. **Distributed actors** — many parallel processes collect experience. 2. **Central learner** — single GPU learner processes batches. 3. **V-trace** — truncated importance sampling corrects off-policy data. 4. **CNN + LSTM** — processes visual observations with temporal memory. 5. **Asynchronous updates** — actors periodically sync with learner.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Many actor processes run the policy in parallel, collecting trajectories. A central learner trains on batches of traject...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: A3C (asynchronous), A2C. Successor: Seed RL, Agent57.

## References

- Espeholt et al. 2018
