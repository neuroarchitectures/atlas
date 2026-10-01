# Mixture Density Network

## Design Philosophy

Instead of predicting a single value, predict a mixture of Gaussian distributions. This enables modeling multi-modal outputs, which is essential for tasks with inherent ambiguity (e.g., inverse problems).

## Functionality

Network outputs: alpha (mixture weights, softmax), mu (means), sigma (stds, softplus). Output distribution: p(y|x) = sum_k alpha_k * N(y; mu_k, sigma_k^2). Loss: negative log-likelihood.

## Used By

MDN | Inverse problems | Trajectory prediction | Speech synthesis

## Features

- **Multi-modal**: Can represent multiple possible outputs.
- **Probabilistic**: Outputs a distribution, not a point estimate.
- **Flexible**: Number of mixture components is a hyperparameter.

## Evolution

Predecessor: MLP (single output). Successor: Conditional VAE, diffusion models.
