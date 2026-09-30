# Architecture: DDIM

## Motivation

DDPM requires hundreds to thousands of denoising steps for high-quality samples, making inference very slow. DDIM addresses this by defining a non-Markovian forward process that shares the same marginal distributions as DDPM but allows deterministic reverse sampling with far fewer steps.

## Core Idea

Define a family of non-Markovian forward diffusion processes that share the same training objective as DDPM. This allows deterministic reverse sampling where fewer steps (e.g., 20-50 vs 1000) produce high-quality samples. The key is that the reverse process can skip steps by jumping directly via the learned noise prediction.

## Architecture

### Overview

![ddim architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | x_T (Noise) | `input` | shape: [1, 3, 64, 64] |
| 2 | DDIM Step t→t-1 | `custom` | type: ddim_step, eta: 0, steps: 50 |
| 3 | ε_θ Network | `custom` | type: ddpm_network |
| 4 | Predict x_0 | `custom` | type: predict_x0 |
| 5 | Direction to x_{t-1} | `custom` | type: direction |
| 6 | Add σ_t noise (if eta>0) | `custom` | type: stochastic_noise, eta: 0 |
| 7 | x_{t-1} | `custom` | type: ddim_update |
| 8 | Loop (T→0) | `custom` | type: reverse_loop |
| 9 | x_0 (Sample) | `output` |  |

</details>

Define a family of non-Markovian forward diffusion processes that share the same training objective as DDPM. This allows deterministic reverse sampling where fewer steps (e.g., 20-50 vs 1000) produce high-quality samples. The key is that the reverse process can skip steps by jumping directly via the learned noise prediction.

### Components

2. **DDIM Step t→t-1** (`custom`, scope: `sampler`) — Params: type: ddim_step, eta: 0, steps: 50
3. **ε_θ Network** (`custom`, scope: `sampler`) — Params: type: ddpm_network
4. **Predict x_0** (`custom`, scope: `sampler`) — Params: type: predict_x0
5. **Direction to x_{t-1}** (`custom`, scope: `sampler`) — Params: type: direction
6. **Add σ_t noise (if eta>0)** (`custom`, scope: `sampler`) — Params: type: stochastic_noise, eta: 0
7. **x_{t-1}** (`custom`, scope: `sampler`) — Params: type: ddim_update
8. **Loop (T→0)** (`custom`, scope: `sampler`) — Params: type: reverse_loop

### Data Flow

The architecture processes input through a sequence of 9 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to DDIM.

## Evolution

DDIM is a direct successor to DDPM, enabling fast deterministic sampling. It influenced Latent Diffusion, DPM-Solver, and other accelerated sampling methods. The deterministic nature also enables interpolation in latent space and consistency model training.

## Source

- **Paper:** arXiv:2010.02502
- **Year:** 2020
- **Authors:** Song et al.
