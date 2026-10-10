# DDPM Noise Schedule

## Design Philosophy

Control the amount of noise added at each timestep in the diffusion process. The schedule (linear, cosine, etc.) determines how quickly information is destroyed and is critical for sample quality.

## Functionality

Beta schedule: beta_1, ..., beta_T (0 < beta_t < 1). Alpha_t = 1 - beta_t. Alpha_bar_t = prod_{s=1}^{t} alpha_s. Forward: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon. Common schedules: linear, cosine, sigmoid.

## Used By

DDPM | Stable Diffusion | Most diffusion models

## Features

- **Flexible**: Linear, cosine, sigmoid schedules.
- **Critical**: Quality depends heavily on schedule choice.
- **Analytical**: Forward process is closed-form.

## Evolution

Predecessor: Langevin dynamics. Successor: Learned schedules, flow matching schedules.
