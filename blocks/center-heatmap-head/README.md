# Anchor-Free Center Heatmap Head

## Design Philosophy

Anchor matching is a brittle hand-tuned assignment. Instead, treat detection as keypoint estimation: a heatmap whose local peaks are object centers, with per-peak regression branches for size, orientation, and velocity — no anchors, no IoU thresholds.

## Functionality

- Head: conv layers → 1-channel heatmap trained with focal-style penalty-reduced loss; peaks via 3×3 max-pool NMS.
- Parallel branches regress 3D size, sin/cos orientation, velocity at each peak; optional second stage refines using point features at box faces.

## Used By

| Model | Role |
|-------|------|
| CenterPoint | BEV 3D object detection over voxel/PointPillars backbone; two-stage refinement variant |

## Features

- **Anchor-free, NMS via peak picking** — assignment-free training.
- **Attribute heads at the peak** — extends trivially to velocity (tracking) and orientation.

## Evolution

- **Predecessor**: CenterNet (2D), CornerNet (keypoint pairs).
- **Successor**: query-based heads (DETR/RT-DETR) replace the heatmap with learned queries.
