# Quantile Embedding (IQN)

## Design Philosophy

A single expected value `Q(s,a)` hides the risk and spread of returns. The implicit quantile network represents the full return distribution and lets the agent optimise any risk measure. The key trick is a cheap cosine embedding of the sampled quantile level `tau` that conditions the value network.

## Functionality

Sample quantile levels `tau ~ U(0,1)` and embed each into a feature vector via cosine basis functions:

```
phi_j(tau) = ReLU( sum_{i=0}^{n-1} cos(pi * i * tau) * w_{ij} + b_j )
```

The embedding is combined with the state features (typically an element-wise product) and fed to a head that outputs `Z_tau(s, a)` — the `tau`-th quantile of the return. Training minimises the quantile Huber loss.

## Used By

IQN

## Features

- **Full distribution**: approximates any quantile, not just the mean.
- **Cheap conditioning**: a small cosine embedding of `tau`.
- **Risk control**: choose risk-sensitive return at inference.
- **Distributional RL**: extends C51 to implicit quantiles.

## Evolution

Predecessor: DQN (expected value), C51 / QR-DQN (discrete distributional RL). Successor: distributional agents with quantile embeddings (IQN, FQF).
