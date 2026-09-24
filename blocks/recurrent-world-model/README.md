# Recurrent World Model + Imagination

## Design Philosophy

Interacting with a real environment is slow and dangerous — so learn a recurrent latent dynamics model that predicts action outcomes, and train the actor/critic *inside imagined rollouts* of that model rather than in the environment. The agent's policy learning is entirely synthetic.

## Functionality

- World model: recurrent latent state (RSSM) reconstructs observations and predicts rewards/continuation from (state, action).
- Actor-critic trained on imagined latent rollouts (value estimation through the latent dynamics); DreamerV3 shows one hyperparameter set across 150+ tasks.

## Used By

| Model | Role |
|-------|------|
| DreamerV3 | World model + critic + actor as three networks; symlog predictions, twohot encoding |

## Features

- **Sample efficiency** — experience is amplified through imagination.
- **Task-agnostic recipe** — the "one config" achievement of DreamerV3.

## Evolution

- **Predecessor**: World Models (Ha & Schmidhuber), Dreamer v1/v2.
- **Successor**: action-conditioned-latent-dynamics (V-JEPA 2-AC) — frozen perception latents + flow-style prediction replace the learned RSSM representation.
