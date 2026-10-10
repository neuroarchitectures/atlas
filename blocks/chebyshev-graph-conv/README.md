# Chebyshev Graph Convolution

## Design Philosophy

Spectral graph convolutions are expensive because they need the Laplacian eigendecomposition. Approximating the spectral filter with a truncated Chebyshev polynomial of the scaled Laplacian avoids eigen-decomposition entirely and localises the filter to a K-hop neighbourhood.

## Functionality

Rescale the Laplacian to `L~ = 2L/lambda_max - I` (spectrum in [-1, 1]) and approximate the filter as

```
g * x ~= sum_{k=0}^{K} theta_k * T_k(L~) * x
```

with the recurrence `T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)`, `T_0 = x`, `T_1 = L~ x`. Each `T_k` is a sparse (edge-local) operation, so no dense matrix is formed.

## Used By

ChebNet

## Features

- **No eigendecomposition**: only sparse matmuls with `L~`.
- **K-hop localisation**: order K controls the receptive field.
- **Linear in edges**: sparse `O(|E|)` per iteration.
- **Generalises GCN**: K=1 with a renormalisation trick reduces to the standard GCN layer.

## Evolution

Predecessor: spectral CNN (Bruna et al., needs eigendecomposition). Successor: GCN (ChebNet simplified to K=1), then attention-based graph layers (GAT).
