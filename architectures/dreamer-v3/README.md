# DreamerV3

## Overview

Dreamer is a **general** algorithm that outperforms **specialized expert algorithms** across a wide range of domains **while using fixed hyperparameters**, making reinforcement learning readily applicable to new problems. The basis is learning a **world model** that gives the agent rich perception and the ability to imagine the future. Although intuitively appealing, robustly learning and leveraging world models had been an open problem.

- **Year:** 2023
- **Authors:** Hafner et al. (Google DeepMind, University of Toronto)
- **Source:** arXiv:2301.04104 — *Mastering Diverse Domains through World Models*
- **Category:** RL/World Model

## Key Characteristics

1. **World model predicts outcomes of potential actions** — equips the agent with rich perception and future imagination.
2. **Critic neural network** — judges the value of each imagined outcome.
3. **Actor neural network** — chooses actions to reach the best outcomes.
4. **Robustness techniques** — a range of techniques based on **normalization, balancing, and transformations**; these are what let the world-model idea actually work, overcoming a previously open problem.
5. **Fixed hyperparameters across over 150 tasks** — robust learning is observed not only across the domains, but also **across model sizes and training budgets**, offering a predictable way to increase performance.
6. **Scaling behaviour made explicit** — **larger model sizes not only achieve higher scores but also require less interaction** to solve a task.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Hafner_et_al._2023_2301.04104.md`](references/papers/Hafner_et_al._2023_2301.04104.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (Dreamer V1/V2, model-based RL, latent imagination → DreamerV3; siblings: world-model and planning agents; contrast: specialized expert algorithms per domain, which DreamerV3 outperforms using one fixed hyperparameter set).
