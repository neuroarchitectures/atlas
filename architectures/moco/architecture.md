# Architecture: MoCo

## Motivation

SimCLR uses in-batch negatives, limiting the number of negative samples. MoCo maintains a large queue of negative features updated by a momentum encoder, enabling unlimited negatives.

## Core Idea

Two encoders: a query encoder (updated by gradient) and a key encoder (updated by momentum). The key encoder produces keys stored in a FIFO queue. Contrastive loss uses the query against the queue.

## Architecture

### Overview

![moco architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | Query View | `conv` | augmentation |
| 2 | Key View | `conv` | different augmentation |
| 3 | Query Encoder | `linear` | updated by gradient descent |
| 3 | Key Encoder (Momentum) | `linear` | updated by momentum (EMA) |
| 4 | Queue | `identity` | 65536 negative keys, FIFO |
| 5 | InfoNCE Loss | `output` | q vs positive k + queue negatives |

</details>

### Components

1. **Query encoder** — Updated by standard gradient descent. 2. **Key encoder** — Updated by exponential moving average (momentum) of the query encoder: theta_k = m * theta_k + (1-m) * theta_q. 3. **Queue** — A FIFO queue of 65536 key features, providing a large and consistent set of negatives. 4. **InfoNCE loss** — Contrastive loss: -log(exp(q . k+ / tau) / sum exp(q . k_i / tau)).

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Two encoders: a query encoder (updated by gradient) and a key encoder (updated by momentum). The key encoder produces keys stored in a FIFO queue. Contrastive loss uses the query against the queue.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: SimCLR, InstDisc. Successor: MoCo v2/v3, DINO.

## References

- He et al. 2020
