# Flow Matching ODE

## Design Philosophy

Instead of the discrete diffusion process, learn a continuous-time ODE that transports samples from a simple distribution (Gaussian) to the data distribution. The ODE is trained via a flow-matching loss.

## Functionality

Learn a velocity field v(x, t) such that dx/dt = v(x, t) transports noise to data. Loss: ||v(x, t) - (x_1 - x_0)||^2 where x_0 ~ noise, x_1 ~ data, t ~ U(0,1). At inference, solve the ODE with Euler or RK4.

## Used By

Flow Matching | Rectified Flow | Stable Diffusion 3 | Voicebox

## Features

- **Continuous**: No discrete timesteps.
- **Optimal transport**: Straight-line paths are more efficient.
- **Simulation-free training**: No need to solve ODE during training.

## Evolution

Predecessor: DDPM (discrete). Successor: Rectified flow, stochastic interpolants.
