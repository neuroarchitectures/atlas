# Architecture: SqueezeNet

## Motivation

Large models (AlexNet, VGG) have millions of parameters. SqueezeNet achieves comparable accuracy with 1MB of parameters by using squeeze (1x1) then expand (1x1 + 3x3) fire modules.

## Core Idea

Fire module: squeeze layer (1x1 conv reducing channels) then expand layer (parallel 1x1 and 3x3 convs). This reduces parameters while maintaining representational power.

## Architecture

### Overview

![squeezenet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | Fire Module 1 | `conv` | squeeze 16, expand 64+32 |
| 3 | Fire Module 2 | `conv` | squeeze 16, expand 64+32 |
| 4 | Output | `output` |  |

</details>

### Components

1. **Squeeze layer** — 1x1 convolution that reduces the number of input channels (e.g., 96 -> 16). 2. **Expand layer** — Parallel 1x1 and 3x3 convolutions that increase channels (e.g., 16 -> 64+32=96). 3. **Concatenation** — The 1x1 and 3x3 outputs are concatenated channel-wise. 4. **Strategies** — Delayed downsampling (larger feature maps early) and deep+wide fire modules improve accuracy.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Fire module: squeeze layer (1x1 conv reducing channels) then expand layer (parallel 1x1 and 3x3 convs). This reduces parameters while maintaining representational power.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: AlexNet. Successor: SqueezeNext, MobileNet (depthwise approach).

## References

- Iandola et al. 2016
