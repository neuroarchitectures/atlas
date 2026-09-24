# Spatial Pyramid Pooling — Fast (SPPF)

## Design Philosophy

Multi-scale receptive fields at the network top help detection, but the classic SPP's pyramid of large max-pools is slow. SPPF gets the same effect by *chaining three small 5×5 max-pools* — each extra pool widens the effective kernel (5→9→13), so the concatenation reproduces the 5/9/13 pyramid with three fast operations.

## Functionality

- `Concat(MaxPool5x5(x), MaxPool5x5(MaxPool5x5(x)), MaxPool5x5(MaxPool5x5(MaxPool5x5(x))))` → 1×1 conv fusion.

## Used By

| Model | Role |
|-------|------|
| YOLOv11 | Placed after the backbone before the neck (P5 level) |

## Features

- **Equivalent receptive fields, fewer ops** than large-kernel pooling.
- **Resolution-independent** — works on any feature map size.

## Evolution

- **Predecessor**: SPP (He et al. 2014) and YOLOv3's SPP module.
- **Related**: aspp-decoder (dilated context instead of pooled context).
