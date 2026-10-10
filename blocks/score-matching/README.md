# Score Matching

## Design Philosophy

Learn the score function (gradient of the log-density) of the data distribution. The score can be used to generate samples via Langevin dynamics: x <- x + eps * nabla log p(x) + sqrt(2*eps) * noise.

## Functionality

Loss: E[||s_theta(x) - nabla_x log p(x)||^2], where s_theta is the score network. In practice, use denoising score matching: add noise to data and predict the noise direction.

## Used By

Score-based generative models | NCSN | Score SDE

## Features

- **No normalizing constant**: Avoids computing Z in p(x) = exp(-E(x)) / Z.
- **Langevin sampling**: MCMC using the score function.
- **Noise-conditional**: Score is conditioned on noise level (NCSN).

## Evolution

Predecessor: Energy-based models. Successor: Score SDE, DDPM (equivalent formulation).
