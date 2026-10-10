# Architecture: NFM

## Motivation

Neural Factorization Machine: bi-interaction pooling layer + deep MLP for factorized feature interactions.

## Core Idea

Neural Factorization Machine: bi-interaction pooling layer + deep MLP for factorized feature interactions.

## Architecture

### Overview

![nfm architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Sparse Features | `input` | shape=[100] |
| 2 | Embedding Layer | `embedding` | vocabSize=1000, dim=16 |
| 3 | Bi-Interaction Pooling | `linear` | outFeatures=16 |
| 4 | Deep MLP | `linear` | outFeatures=1 |
| 5 | Prediction | `output` | — |

</details>

### Components

1. **Sparse Features** — input layer. 2. **Embedding Layer** — embedding layer. 2. **Bi-Interaction Pooling** — linear layer. 2. **Deep MLP** — linear layer. 2. **Prediction** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Neural Factorization Machine: bi-interaction pooling layer + deep MLP for factorized feature interactions.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
