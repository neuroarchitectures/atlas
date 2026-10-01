# Architecture: MLP-Mixer

## Motivation

Vision architectures rely on attention (ViT) or convolution (CNN). MLP-Mixer shows that pure MLPs can match performance if spatial mixing and channel mixing are separated.

## Core Idea

Two types of MLPs per block: (1) Token-mixing MLP that operates across spatial locations (transposed), (2) Channel-mixing MLP that operates per-location. No attention, no convolution.

## Architecture

### Overview

![mlp-mixer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image Patches | `input` | shape: [196, 256] (14x14 patches, 256 dim) |
| 2 | LayerNorm | `layerNorm` |  |
| 3 | Token-Mixing MLP | `linear` | mixes across 196 spatial tokens |
| 4 | GELU | `gelu` |  |
| 5 | Residual | `add` |  |
| 6 | LayerNorm 2 | `layerNorm` |  |
| 7 | Channel-Mixing MLP | `linear` | mixes across 256 channels |
| 8 | GELU 2 | `gelu` |  |
| 9 | Output | `output` |  |

</details>

### Components

1. **Token-mixing MLP** — Operates on the transposed input (spatial dimension). Each column is an independent spatial location; the MLP mixes information across locations. 2. **Channel-mixing MLP** — Operates per-location, mixing channels. 3. **No attention** — All information routing is done by MLPs, which are position-independent. 4. **Equal parameter count** — Token-mixing MLP has O(S^2) parameters (S=patches), channel-mixing has O(C^2).

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Two types of MLPs per block: (1) Token-mixing MLP that operates across spatial locations (transposed), (2) Channel-mixing MLP that operates per-location. No attention, no convolution.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ViT. Successor: gMLP, ResMLP, CycleMLP, ConvMixer.

## References

- Tolstikhin et al. 2021
