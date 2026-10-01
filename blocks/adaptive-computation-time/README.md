# Adaptive Computation Time (ACT)

## Design Philosophy

Allow the network to dynamically decide how many layers to process per input. Easy inputs can exit early, hard inputs use more layers. This enables adaptive computation budgets.

## Functionality

Each layer outputs a halting probability p_t. Accumulate: R_t = R_{t-1} + p_t. When R_t >= 1 - epsilon, stop. The output is a weighted sum of layer outputs, weighted by p_t. A pondering loss encourages efficiency.

## Used By

Universal Transformers | Adaptive depth models | Neural program interpreters

## Features

- **Adaptive depth**: Different inputs use different numbers of layers.
- **Halting distribution**: p_t sums to ~1, ensuring proper weighting.
- **Pondering cost**: Regularizer that penalizes excessive computation.

## Evolution

Predecessor: Fixed-depth networks. Successor: Mixture of Depths, early-exit transformers.
