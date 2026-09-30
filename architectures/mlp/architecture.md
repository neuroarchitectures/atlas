# Architecture: MLP

## Motivation

A single perceptron can only learn linearly separable functions. The MLP overcomes this by stacking multiple layers with nonlinear activations, enabling the learning of arbitrary decision boundaries. Backpropagation provides an efficient algorithm for training these multi-layer networks.

## Core Idea

Compose multiple dense layers with nonlinear activations: h = σ(Wx + b). Each layer transforms the representation, and the composition of nonlinear functions can approximate any continuous function (universal approximation theorem). Backpropagation computes gradients via the chain rule.

## Architecture

### Overview

![mlp architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [1, 784] |
| 2 | Hidden Layer 1 | `linear` | outFeatures: 256, inFeatures: 784 |
| 3 | ReLU | `relu` |  |
| 4 | Hidden Layer 2 | `linear` | outFeatures: 128, inFeatures: 256 |
| 5 | ReLU | `relu` |  |
| 6 | Output Layer | `linear` | outFeatures: 10, inFeatures: 128 |
| 7 | Softmax | `softmax` |  |
| 8 | Output | `output` |  |

</details>

Compose multiple dense layers with nonlinear activations: h = σ(Wx + b). Each layer transforms the representation, and the composition of nonlinear functions can approximate any continuous function (universal approximation theorem). Backpropagation computes gradients via the chain rule.

### Components

2. **Hidden Layer 1** (`linear`, scope: `layer.0`) — Params: outFeatures: 256, inFeatures: 784
3. **ReLU** (`relu`, scope: `layer.0`) — Params: none
4. **Hidden Layer 2** (`linear`, scope: `layer.1`) — Params: outFeatures: 128, inFeatures: 256
5. **ReLU** (`relu`, scope: `layer.1`) — Params: none
6. **Output Layer** (`linear`, scope: `output`) — Params: outFeatures: 10, inFeatures: 128
7. **Softmax** (`softmax`, scope: `output`) — Params: none

### Data Flow

The architecture processes input through a sequence of 8 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to MLP.

## Evolution

The MLP is the foundational deep architecture, introducing deep composition with backpropagation. Successors: CNN (spatial MLP), RNN (temporal MLP), ResNet (MLP + residual), Transformer (MLP + attention), and all modern architectures. MLPs are still used as FFN blocks within Transformers.

## Source

- **Paper:** Nature 1986
- **Year:** 1986
- **Authors:** Rumelhart et al.
