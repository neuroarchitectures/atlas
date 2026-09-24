# RoIAlign

## Design Philosophy

Extract fixed-size features from a region of interest (RoI) using **bilinear interpolation** at exact continuous coordinates, avoiding the quantization misalignment of RoIPool. The philosophy: RoIPool rounds RoI coordinates to integers, causing misalignment that's tolerable for bounding boxes but *kills* pixel-level mask accuracy. RoIAlign computes exact input-feature locations, preserving spatial precision.

## Functionality

- **Input**: A feature map and a set of RoIs (continuous coordinates `[x1, y1, x2, y2]`).
- **For each RoI**: Divide into a `k×k` (e.g., 7×7) grid of sampling points.
- **Bilinear interpolation**: At each sampling point (continuous coordinates), interpolate the feature map's 4 nearest grid points.
- **Output**: A `k×k×C` feature per RoI, exactly aligned to the input feature map.
- No quantization anywhere in the path.

## Used By

| Model | Role |
|-------|------|
| Mask R-CNN | Critical for mask quality (improves mask AP by 1–3 points) |
| Faster R-CNN (v2) | Improved RoI extraction |
| Cascade R-CNN | RoIAlign for multi-stage detection |

## Features

- **Exact alignment**: No quantization → pixel-level precision.
- **Mask-quality critical**: The difference between usable and unusable instance masks.
- **Drop-in for RoIPool**: Same interface, better gradients.

## Evolution

- **Predecessor**: RoIPool (quantized, misaligned); SPP (spatial pyramid pooling).
- **Successor**: AlignDETR / DETR-class architectures remove RoIs entirely (set-based prediction); but RoIAlign remains in two-stage detectors.
