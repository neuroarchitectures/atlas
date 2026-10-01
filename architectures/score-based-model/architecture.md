# Architecture: Score-based Generative Model

## Motivation

Diffusion models (DDPM) learn to denoise but are tied to a specific noise schedule. Score-based models generalize this by directly estimating the score function, enabling continuous-time sampling via SDEs.

## Core Idea

Train a network s_theta(x, t) to estimate the score (nabla log p_t(x)) at various noise levels t. Generate samples by solving the reverse-time SDE using the estimated score.

## Architecture

### Overview

![score-based generative model architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Noisy Input x_t | `input` | shape: [3, 32, 32] |
| 2 | Time Embedding | `embedding` | numEmbeddings: 1000, embeddingDim: 128 |
| 3 | Score Network (UNet) | `conv` | estimates nabla log p_t(x_t) |
| 4 | Score nabla log p | `output` | used in Langevin/SDE sampling |

</details>

### Components

1. **Score network** — A UNet-based network s_theta(x, t) that estimates the score (gradient of log density) at noise level t. 2. **Time embedding** — Encodes the continuous noise level t into a vector. 3. **Reverse SDE** — Sampling solves dx = -[f(x,t) - g(t)^2 s_theta(x,t)] dt + g(t) dw using the estimated score. 4. **Score matching loss** — Denoising score matching: ||s_theta(x_t, t) - nabla log p(x_t|x_0)||^2.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Train a network s_theta(x, t) to estimate the score (nabla log p_t(x)) at various noise levels t. Generate samples by solving the reverse-time SDE using the estimated score.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DDPM (discrete). Successor: Score SDE (continuous), Flow Matching, Rectified Flow.

## References

- Song & Ermon 2019
