# Architecture: Spatial Transformer Network

## Motivation

CNNs have limited translation invariance (only from pooling). STN introduces a differentiable spatial transformer that can learn to align features, improving invariance to geometric transformations.

## Core Idea

Three components: (1) Localisation network predicts transformation parameters theta, (2) Grid generator creates a sampling grid based on theta, (3) Sampler uses bilinear interpolation to sample from the input.

## Architecture

### Overview

![spatial transformer network architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (5 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Feature Map | `input` | shape: [32, 16, 16] |
| 2 | Localisation Net | `linear` | predicts 6 affine params (theta) |
| 3 | Grid Generator | `identity` | creates sampling grid from theta |
| 4 | Bilinear Sampler | `identity` | differentiable sampling |
| 5 | Transformed Map | `output` | spatially transformed feature map |

</details>

### Components

1. **Localisation network** — A small CNN or FC network that predicts the 6 parameters of an affine transformation matrix theta. 2. **Grid generator** — Creates a regular grid and transforms it by theta to get the sampling locations. 3. **Bilinear sampler** — Samples the input feature map at the grid locations using bilinear interpolation. The gradient flows through the sampling to the localisation network. 4. **Pluggable** — STN can be inserted anywhere in a CNN.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Three components: (1) Localisation network predicts transformation parameters theta, (2) Grid generator creates a sampling grid based on theta, (3) Sampler uses bilinear interpolation to sample from t
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: CNN. Successor: Deformable Convolution Networks (DCN).

## References

- Jaderberg et al. 2015
