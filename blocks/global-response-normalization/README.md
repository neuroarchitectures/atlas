# Global Response Normalization (GRN)

## Design Philosophy

In ConvNeXt V2, many channels die out during masked-autoencoder pre-training because nothing encourages channels to compete. GRN adds a lightweight, parameter-free-ish unit that lets channels compete globally across the spatial and batch dimensions, reviving feature diversity.

## Functionality

Compute a per-channel global (spatial) `L2` response, divide it by the mean response across channels to get a relative importance, then re-weight the input:

```
G(X) = ||X_c||  over spatial dims           (per channel)
N(X) = G(X) / mean_c G(X)                   (relative importance)
GRN(X) = gamma * (N(X) * X) + beta + X      (residual)
```

`gamma` and `beta` are learnable per-channel vectors initialised to zero.

## Used By

ConvNeXt V2

## Features

- **Channel competition**: boosts informative channels, suppresses redundant ones.
- **Contrast without entropy**: works even for single-layer feature maps (unlike divisive norm).
- **Two parameters**: negligible cost.
- **Residual**: initialised as identity, safe to insert anywhere.

## Evolution

Predecessor: LayerNorm/BatchNorm (no cross-channel competition), divisive normalization. Used by: ConvNeXt V2 FCMAE pre-training. Sibling: Squeeze-and-Excite (excitation via gating).
