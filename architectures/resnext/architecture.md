# Architecture: ResNeXt

## Motivation

ResNet improved depth but not width. ResNeXt introduces 'cardinality' (number of parallel paths) as a new dimension for scaling, showing it's more effective than depth or width.

## Core Idea

Use grouped convolutions to split the bottleneck block into C independent paths (cardinality=32). Each path has 4 channels. All paths are summed and added via residual connection.

## Architecture

### Overview

![resnext architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [256, 56, 56] |
| 2 | 1x1 Conv (reduce) | `conv` | outChannels: 128, bottleneck |
| 3 | 3x3 GroupConv | `conv` | 32 groups, 4 channels each |
| 4 | 1x1 Conv (expand) | `conv` | outChannels: 256 |
| 5 | Residual | `add` |  |
| 6 | Output | `output` |  |

</details>

### Components

1. **Grouped convolution** — The 3x3 convolution is split into 32 groups, each with 4 input and 4 output channels. This is equivalent to 32 parallel paths. 2. **Bottleneck** — 1x1 reduce -> 3x3 grouped -> 1x1 expand. 3. **Cardinality** — The number of groups (C=32) is a key hyperparameter, more effective than depth or width for the same parameter count. 4. **Aggregated residual transformation** — All paths are summed, then added to the input via residual connection.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Use grouped convolutions to split the bottleneck block into C independent paths (cardinality=32). Each path has 4 channels. All paths are summed and added via residual connection.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ResNet, Inception (multi-branch). Successor: ConvNeXt (uses depthwise conv, extreme cardinality).

## References

- Xie et al. 2017
