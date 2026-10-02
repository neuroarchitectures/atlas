# Architecture: xDeepFM

## Motivation

Compressed interaction network (CIN) + deep network for explicit and implicit high-order feature interactions.

## Core Idea

Compressed interaction network (CIN) + deep network for explicit and implicit high-order feature interactions.

## Architecture

### Overview

![xdeepfm architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Sparse Features | `input` | shape=[100] |
| 2 | Embedding Layer | `embedding` | vocabSize=1000, dim=16 |
| 3 | CIN (Compressed Interaction Network) | `linear` | outFeatures=128 |
| 4 | Deep Network (DNN) | `linear` | outFeatures=128 |
| 5 | Concat (CIN + Deep) | `linear` | outFeatures=1 |
| 6 | Prediction | `output` | — |

</details>

### Components

1. **Sparse Features** — input layer. 2. **Embedding Layer** — embedding layer. 2. **CIN (Compressed Interaction Network)** — linear layer. 2. **Deep Network (DNN)** — linear layer. 2. **Concat (CIN + Deep)** — linear layer. 2. **Prediction** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Compressed interaction network (CIN) + deep network for explicit and implicit high-order feature interactions.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
