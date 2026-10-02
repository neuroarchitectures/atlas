# Architecture: ControlNet

## Motivation

Diffusion models generate diverse images but lack control over spatial structure. Users want to specify edges, depth maps, or poses.

## Core Idea

Create a trainable copy of the diffusion model's encoder blocks, connected to the original via zero convolution layers. The copy learns to condition on spatial inputs while the original model is frozen.

## Architecture

### Overview

![controlnet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Latent | `input` | shape: [4, 64, 64] |
| 2 | Condition Image | `input` | shape: [3, 512, 512] |
| 3 | Zero Conv Control | `control-block` |  |
| 4 | UNet (Frozen Copy) | `unet` |  |
| 5 | Merge Features | `add` |  |
| 6 | Denoised Latent | `output` |  |

</details>

### Components

1. **Condition encoder** — Processes spatial condition (Canny edges, depth, pose). 2. **Trainable copy** — Clone of UNet encoder blocks. 3. **Zero convolution** — 1x1 convs initialized to zero, gradually learning to inject conditions. 4. **Feature merging** — Control features added to original UNet features.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Create a trainable copy of the diffusion model's encoder blocks, connected to the original via zero convolution layers. 
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Stable Diffusion. Successor: T2I-Adapter, ControlNet-XS.

## References

Zhang & Agrawala 2023
