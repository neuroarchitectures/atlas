# Personalized PageRank Propagation (APPNP)

## Design Philosophy

Stacking GCN layers entangles feature learning with propagation, causing oversmoothing as depth grows. APPNP decouples the two: learn features with an MLP, then spread them with a fixed, personalised PageRank diffusion that can reach far without extra parameters.

## Functionality

Predict per-node features `H = MLP(X)`, then iterate the personalised PageRank diffusion:

```
Z^{(0)} = H
Z^{(k+1)} = (1 - alpha) * A_hat * Z^{(k)} + alpha * H
```

where `A_hat` is the normalised adjacency and `alpha` is the teleport probability that keeps each node anchored to its own prediction. After `K` steps, `Z^{(K)}` is the output.

## Used By

APPNP

## Features

- **Decoupled**: feature transform and propagation are separate.
- **No oversmoothing**: teleport keeps node identity.
- **Parameter-free propagation**: only the MLP learns.
- **Long-range**: large `K` gives a wide diffusion without adding layers.

## Evolution

Predecessor: GCN (coupled propagation + transform). Sibling: LightGCN (pure propagation), graph diffusion convolution. Successor: decoupled GNNs (SGC, APPNP, PPRGo).
