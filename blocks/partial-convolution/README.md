# Partial Convolution (PConv)

## Design Philosophy

Convolutional layers exhibit strong spatial redundancy, so computing *every* channel at every position is wasteful. PConv applies regular dense convolution to only a subset of channels and passes the rest through untouched — optimizing FLOPS *and* memory access (the real bottleneck on edge devices), not just FLOPs.

## Functionality

- Dense 3×3 conv on the first `r·C` channels (`r = 1/4` typical), identity on the remainder.
- Followed by a pointwise conv to mix channel information across the split.

## Used By

| Model | Role |
|-------|------|
| FasterNet | PConv + pointwise conv per block, 4-stage FasterNet stack |

## Features

- **Lower memory access** than depthwise conv per unit of computation (many channels touched at once, few positions).
- **Keeps regular conv hardware paths** — no exotic depthwise scheduling.

## Evolution

- **Predecessor**: depthwise-separable-conv, ghost-module — other factorizations of the conv cost.
- **Successor**: FasterNet V2; conceptually related to Faster R-CNN's partial usage of features (unrelated naming).
