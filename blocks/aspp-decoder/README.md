# Atrous Spatial Pyramid Pooling (ASPP)

## Design Philosophy

Objects of the same class appear at wildly different scales. Probe the features with parallel *dilated* convolutions at several dilation rates — a spatial pyramid without resolution loss — so each location is described by multi-scale context simultaneously.

## Functionality

- Parallel branches: 1×1 conv + several 3×3 convs with rates {6, 12, 18} (+ image-level global pooling branch).
- Concatenate → 1×1 conv fusion → per-pixel classifier.
- LR-ASPP (MobileNetV3): fewer branches, reduced rates, SE augmentation, for mobile segmentation.

## Used By

| Model | Role |
|-------|------|
| MobileNetV3 (segmentation variants) | LR-ASPP decoder head with SE module |

## Features

- **Multi-scale context at full resolution** — no strided pyramid.
- **Dilation choice is the scale knob** — tunable per dataset resolution.

## Evolution

- **Predecessor**: DeepLab v2 ASPP; SPP (spatial pyramid pooling, 2014).
- **Related**: SPPF (YOLO) is the detection-side fast variant; feature-pyramid-network is the strided counterpart.
