# Architecture: DenseNet

## Motivation

ResNet uses additive skip connections. DenseNet concatenates feature maps from all previous layers, enabling feature reuse, stronger gradient flow, and fewer parameters.

## Core Idea

Each layer receives the concatenation of all preceding layers' feature maps. Dense blocks of L layers each have L(L+1)/2 direct connections. Transition layers (1x1 conv + avg pool) reduce dimensions between blocks.

## Architecture

### Overview

![densenet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3x224x224 |
| 2 | Initial Conv | `conv2d` | 7x7, stride 2 |
| 3 | Dense Block 1 | `conv2d` | 6 layers, growth rate 32 |
| 4 | Transition | `conv2d` | 1x1 conv + 2x2 pool |
| 5 | Classification | `output` | 1000 classes |

</details>

### Components

1. **Dense connectivity** — each layer gets all previous feature maps concatenated. 2. **Growth rate** — each layer adds k=32 new feature maps. 3. **Bottleneck** — 1x1 conv before 3x3 to reduce input channels. 4. **Transition layers** — compression + downsampling between blocks. 5. **Feature reuse** — reduces redundant feature maps, fewer params. 6. **Strong gradients** — short paths to all layers.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Each layer receives the concatenation of all preceding layers' feature maps. Dense blocks of L layers each have L(L+1)/2...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ResNet (additive). Successor: EfficientNet (compound scaling).

## References

- Huang et al. 2017
