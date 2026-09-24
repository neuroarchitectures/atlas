# Flow Matching Objective

## Design Philosophy

Diffusion training needs a noise schedule and many steps; flow matching regresses a *velocity field* directly: learn v_ψ(z_t, t) ≈ u_t transporting N(0, I) to the data distribution along a (linear) path — simpler, schedule-free, and exact at inference via the induced transport.

## Functionality

- Sample t, z_t = (1−t) z_0 + t z_1 style paths; loss `‖v_ψ(z_t, t) − u_t‖²` where u_t is the path velocity.
- Inference integrates the ODE (few steps); per-block FM for sequence priors (CLDLM), next-geometry-latent prediction (VGGT-World).

## Used By

| Model | Role |
|-------|------|
| CLDLM | Per-block flow matching on the DiT vector field over latent text blocks |
| VGGT-World | Flow-matching-style next-step prediction of geometry latents — more stable than MSE regression |

## Features

- **No schedule design** — the path defines everything.
- **Stable regression target** for high-dimensional latents where MSE underperforms.

## Evolution

- **Predecessor**: score-based diffusion (DDPM/EDM); rectified flow, stochastic interpolants.
- **Successor**: mean-flow, shortcut models — few-step direct transport.
