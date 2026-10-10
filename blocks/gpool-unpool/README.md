# Graph Pooling / Unpooling (gPool/gUnpool)

## Design Philosophy

CNNs gain much of their power from downsampling: pooling enlarges the receptive field and builds a multi-scale hierarchy. Graph U-Nets bring the same idea to graphs by learning *which* nodes to keep, then restoring them exactly on the upswing.

## Functionality

**gPool** projects node features with a learnable vector `p`, scores each node, keeps the top-k:

```
y = p . x_i / ||p||        (score)
idx = topk(y, k)
x' = x[idx] ;  A' = A[idx, idx]
```

**gUnpool** reverses the operation with the stored `idx`, scattering the pooled features back to their original positions and restoring the edges.

## Used By

Graph U-Net

## Features

- **Learned selection**: pooling follows learnable importance, not a fixed rule.
- **Exact inverse index**: gUnpool is parameter-free and lossless in position.
- **Hierarchical receptive field**: stacking pools grows the neighbourhood.
- **Skip connections**: the stored indices enable U-Net style shortcuts.

## Evolution

Predecessor: graph coarsening (DiffPool, structural matching). Sibling: DiffPool (soft, learned assignment). Successor: top-k / SAGPool variants in graph U-Nets.
