# Neural ODE Block

## Design Philosophy

Instead of a discrete sequence of layers, model the network as a continuous-depth ODE: dz/dt = f(z(t), t). This allows arbitrary depth, memory-efficient backprop via adjoint method, and continuous-time dynamics.

## Functionality

Forward: Solve ODE from t=0 to t=1 using a numerical solver (RK4, Dormand-Prince). Backward: Use the adjoint sensitivity method to compute gradients without storing intermediate activations.

## Used By

Neural ODE | Continuous normalizing flows | Latent ODE for time series

## Features

- **Continuous depth**: No fixed number of layers; depth is determined by the ODE solver.
- **Memory efficient**: Adjoint method uses O(1) memory for backprop.
- **Adaptive computation**: Solver can use more steps for harder inputs.

## Evolution

Predecessor: ResNet (discrete residual). Successor: Augmented Neural ODE, Neural CDE.
