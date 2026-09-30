# Architecture: DDPM

## Motivation

Existing generative models (GANs, VAEs, normalizing flows) have trade-offs: GANs are hard to train, VAEs produce blurry samples, flows require invertible architectures. DDPM shows that diffusion-based models can produce high-quality samples comparable to GANs while being stable to train.

## Core Idea

Define a forward diffusion process that gradually adds Gaussian noise to data over T steps, and learn the reverse process that denoises step by step. The key insight is that training reduces to predicting the noise added at each step, which is a simple regression objective.

## Architecture

### Overview

![ddpm architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (14 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Noisy Image x_t | `input` | shape: [1, 3, 64, 64] |
| 2 | Timestep t | `input` | shape: [1] |
| 3 | Timestep Embedding | `custom` | type: sinusoidal_embedding, dim: 128 |
| 4 | Conv In | `conv2d` | outChannels: 128, kernelSize: 3, stride: 1, padding: 1, inChannels: 3 |
| 5 | ResBlock + Time | `custom` | type: resblock_time, channels: 128, timeDim: 128 |
| 6 | ResBlock + Time | `custom` | type: resblock_time, channels: 128, timeDim: 128 |
| 7 | Downsample | `custom` | type: downsample, channels: 128 |
| 8 | ResBlock + Time | `custom` | type: resblock_time, channels: 128, timeDim: 128 |
| 9 | Self-Attention | `custom` | type: self_attention, channels: 128, heads: 4 |
| 10 | ResBlock + Time | `custom` | type: resblock_time, channels: 128, timeDim: 128 |
| 11 | Upsample | `custom` | type: upsample, channels: 128 |
| 12 | ResBlock + Time | `custom` | type: resblock_time, channels: 128, timeDim: 128 |
| 13 | Conv Out | `conv2d` | outChannels: 3, kernelSize: 3, stride: 1, padding: 1, inChannels: 128 |
| 14 | Predicted Noise ε | `output` |  |

</details>

Define a forward diffusion process that gradually adds Gaussian noise to data over T steps, and learn the reverse process that denoises step by step. The key insight is that training reduces to predicting the noise added at each step, which is a simple regression objective.

### Components

3. **Timestep Embedding** (`custom`, scope: `model`) — Params: type: sinusoidal_embedding, dim: 128
4. **Conv In** (`conv2d`, scope: `model`) — Params: outChannels: 128, kernelSize: 3, stride: 1, padding: 1, inChannels: 3
5. **ResBlock + Time** (`custom`, scope: `down.0`) — Params: type: resblock_time, channels: 128, timeDim: 128
6. **ResBlock + Time** (`custom`, scope: `down.0`) — Params: type: resblock_time, channels: 128, timeDim: 128
7. **Downsample** (`custom`, scope: `down.0`) — Params: type: downsample, channels: 128
8. **ResBlock + Time** (`custom`, scope: `mid`) — Params: type: resblock_time, channels: 128, timeDim: 128
9. **Self-Attention** (`custom`, scope: `mid`) — Params: type: self_attention, channels: 128, heads: 4
10. **ResBlock + Time** (`custom`, scope: `mid`) — Params: type: resblock_time, channels: 128, timeDim: 128
11. **Upsample** (`custom`, scope: `up.0`) — Params: type: upsample, channels: 128
12. **ResBlock + Time** (`custom`, scope: `up.0`) — Params: type: resblock_time, channels: 128, timeDim: 128
13. **Conv Out** (`conv2d`, scope: `model`) — Params: outChannels: 3, kernelSize: 3, stride: 1, padding: 1, inChannels: 128

### Data Flow

The architecture processes input through a sequence of 14 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to DDPM.

## Evolution

DDPM builds on score-based generative models and diffusion processes. Successors: DDIM (fast sampling), Latent Diffusion/Stable Diffusion (diffusion in latent space), DiT (transformer-based denoiser), Flow Matching, and Consistency Models.

## Source

- **Paper:** arXiv:2006.11239
- **Year:** 2020
- **Authors:** Ho et al.
