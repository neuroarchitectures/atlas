# ActNorm (Activation Normalization)

## Design Philosophy

Replace batch normalization in normalizing flows with a data-independent normalization. ActNorm initializes scale and bias from the first batch, then learns them. This avoids the issues of BN in flow models.

## Functionality

y = x * scale + bias, where scale and bias are per-channel learnable parameters. Initialized so that the first batch has zero mean and unit variance per channel.

## Used By

Glow | Flow-based generative models

## Features

- **No batch statistics**: Unlike BN, doesn't use batch statistics at inference.
- **Data-dependent init**: First batch sets initial values.
- **Invertible**: Scale and bias have trivial inverse.

## Evolution

Predecessor: Batch normalization. Successor: Variants in flow models.
