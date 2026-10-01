# Architecture: Deep Belief Network

## Motivation

Training deep networks was intractable before 2006. Hinton showed that pre-training each layer as an RBM, then fine-tuning with backpropagation, could effectively initialize deep networks.

## Core Idea

A DBN is a stack of RBMs where each RBM's hidden layer serves as the visible layer of the next. Greedy layer-wise pre-training initializes weights, followed by discriminative fine-tuning.

## Architecture

### Overview

![deep belief network architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [784] |
| 2 | RBM1 Visible | `linear` | outFeatures: 500, inFeatures: 784 |
| 3 | RBM1 Hidden | `sigmoid` |  |
| 4 | RBM2 Visible | `linear` | outFeatures: 500, inFeatures: 500 |
| 5 | RBM2 Hidden | `sigmoid` |  |
| 6 | RBM3 Visible | `linear` | outFeatures: 2000, inFeatures: 500 |
| 7 | RBM3 Hidden | `sigmoid` |  |
| 8 | Output | `output` |  |

</details>

### Components

1. **Stacked RBMs** — Each layer is pre-trained as an RBM using contrastive divergence. 2. **Greedy layer-wise training** — Each RBM is trained independently, then its hidden activations become the input to the next RBM. 3. **Fine-tuning** — After pre-training, the network is unrolled and fine-tuned with backpropagation for the target task.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A DBN is a stack of RBMs where each RBM's hidden layer serves as the visible layer of the next. Greedy layer-wise pre-training initializes weights, followed by discriminative fine-tuning.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: RBM. Successor: Stacked Autoencoders, deep pre-training strategies.

## References

- Hinton et al. 2006
