# Architecture: RegNet

## Motivation

Manually designing CNN architectures is labor-intensive. Existing design spaces (NASNet, AmoebaNet) are too unconstrained.

## Core Idea

Define a structured design space where model width and depth follow a quantized linear function, enabling efficient search over a constrained, high-quality space.

## Architecture

### Overview

![regnet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | Stem | `conv2d` |  |
| 3 | Stage 1 | `residual-block` | depth: 2 |
| 4 | Stage 2 | `residual-block` | depth: 7 |
| 5 | Stage 3 | `residual-block` | depth: 12 |
| 6 | Stage 4 | `residual-block` | depth: 1 |
| 7 | Global Avg Pool | `pooling` |  |
| 8 | Classifier | `linear` | outFeatures: 1000 |
| 9 | Class Logits | `output` |  |

</details>

### Components

1. **Stem** — Single conv layer. 2. **Four stages** — Each stage has residual blocks with widths following a linear parameterization w_j = w_0 + w_a * j. 3. **Width quantization** — Widths rounded to nearest multiple of a quantization parameter. 4. **Global avg pool + classifier** — Standard classification head.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Define a structured design space where model width and depth follow a quantized linear function, enabling efficient sear
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ResNet, ResNeXt. Successor: EfficientNet, RegNetY.

## References

Radosavovic et al. 2020
