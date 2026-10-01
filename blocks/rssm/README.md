# RSSM (Recurrent State-Space Model)

## Design Philosophy

A state-space model with both deterministic (LSTM) and stochastic components. The deterministic path provides stable long-term memory, while the stochastic path models uncertainty.

## Functionality

1. Deterministic: h_t = LSTM(h_{t-1}, z_{t-1}, a_{t-1}). 2. Stochastic: posterior z_t ~ q(z_t | h_t, o_t) (inference), prior z_hat_t ~ p(z_hat_t | h_t) (prediction). 3. KL divergence between posterior and prior is the representation loss.

## Used By

Dreamer | DreamerV2 | DreamerV3 | World models

## Features

- **Dual state**: Deterministic + stochastic.
- **Posterior vs prior**: Inference model vs dynamics model.
- **Latent imagination**: Can roll forward without observations.

## Evolution

Predecessor: VAE, LSTM. Successor: DreamerV2/V3, world model variants.
