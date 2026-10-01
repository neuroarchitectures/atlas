# Architecture: Neural Process

## Motivation

Gaussian Processes provide calibrated uncertainty but are computationally expensive. Neural Networks are fast but lack uncertainty. Neural Processes combine both: learn a distribution over functions from a context set.

## Core Idea

Condition on a context set (x_i, y_i), encode it into a latent distribution z ~ q(z|context), sample z, then decode to predict y for new x. This amortizes inference across tasks.

## Architecture

### Overview

![neural process architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Context (x,y) | `input` | shape: [10, 2] |
| 2 | Encoder | `linear` | outFeatures: 128, inFeatures: 2 |
| 3 | ReLU | `relu` |  |
| 4 | Aggregate (Mean) | `identity` | mean over context set |
| 5 | Mean | `linear` | outFeatures: 64 |
| 5 | LogVar | `linear` | outFeatures: 64 |
| 6 | Sample z | `identity` | reparameterize |
| 7 | Decoder | `linear` | outFeatures: 1, inFeatures: 65 (z + x_target) |
| 8 | Prediction | `output` |  |

</details>

### Components

1. **Encoder** — Maps each (x_i, y_i) context pair to an embedding. 2. **Aggregator** — Computes the mean of context embeddings to get a permutation-invariant summary. 3. **Latent distribution** — Mean and log-variance of a Gaussian latent z, which represents the function class. 4. **Decoder** — Takes a sampled z and a target x, predicts y with uncertainty.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Condition on a context set (x_i, y_i), encode it into a latent distribution z ~ q(z|context), sample z, then decode to predict y for new x. This amortizes inference across tasks.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Conditional Neural Process. Successor: Attentive Neural Process, Neural Process Family.

## References

- Garnelo et al. 2018
