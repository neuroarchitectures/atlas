# Architecture: ADALINE

## Motivation

The Perceptron (1957) could learn linear decision boundaries but used a step-function loss that was not differentiable. Widrow and Hoff sought a network with a smooth, differentiable learning rule that could converge reliably.

## Core Idea

ADALINE (Adaptive Linear Neuron) uses a linear activation function during training and the LMS (Least Mean Squares) delta rule for weight updates. Unlike the Perceptron, the error is computed before the quantizer, making the loss surface smooth and quadratic.

## Architecture

### Overview

![adaline architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [2] |
| 2 | Linear | `linear` | outFeatures: 1, inFeatures: 2 |
| 3 | Linear Activation | `identity` |  |
| 4 | Sign Quantizer | `threshold` | threshold: 0 |
| 5 | Output | `output` |  |

</details>

### Components

1. **Linear layer** — A single fully-connected layer mapping input features to a scalar output. 2. **Linear (identity) activation** — Unlike the Perceptron's step function, ADALINE uses a linear activation during training, making the loss differentiable everywhere. 3. **Sign quantizer** — Applied only at inference time to produce binary output (+1/-1).

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — ADALINE (Adaptive Linear Neuron) uses a linear activation function during training and the LMS (Least Mean Squares) delta rule for weight updates. Unlike the Perceptron, the error is computed before t
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Perceptron (1957). Successor: MADALINE (1962), multi-layer adaptive networks.

## References

- Widrow & Hoff 1960
