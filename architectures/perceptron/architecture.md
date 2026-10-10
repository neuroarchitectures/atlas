# Architecture: Perceptron

## Motivation

How can a machine learn to classify patterns? Rosenblatt's Perceptron introduced a simple but powerful model: a linear classifier with a binary threshold activation, trained by the perceptron learning rule (update weights on misclassification).

## Core Idea

Compute a weighted sum of inputs and apply a threshold function: y = sign(w·x + b). Training adjusts weights when the perceptron misclassifies: w ← w + α(y_target - y_pred)x. The perceptron convergence theorem guarantees learning for linearly separable data.

## Architecture

### Overview

![perceptron architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input x | `input` | shape: [1, 10] |
| 2 | Weights w | `linear` | outFeatures: 1, inFeatures: 10 |
| 3 | Threshold | `custom` | type: sign_activation, threshold: 0 |
| 4 | Output y | `output` |  |

</details>

Compute a weighted sum of inputs and apply a threshold function: y = sign(w·x + b). Training adjusts weights when the perceptron misclassifies: w ← w + α(y_target - y_pred)x. The perceptron convergence theorem guarantees learning for linearly separable data.

### Components

2. **Weights w** (`linear`, scope: `model`) — Params: outFeatures: 1, inFeatures: 10
3. **Threshold** (`custom`, scope: `model`) — Params: type: sign_activation, threshold: 0

### Data Flow

The architecture processes input through a sequence of 4 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Perceptron.

## Evolution

The Perceptron is the foundational neural architecture. Its limitation (cannot learn XOR) motivated multi-layer perceptrons (MLP) with hidden layers and backpropagation. Successors: ADALINE, MADALINE, MLP, and all modern neural networks.

## Source

- **Paper:** Psychological Review 1958
- **Year:** 1957
- **Authors:** Rosenblatt
