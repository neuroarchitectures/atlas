# Pointmap Regression

## Design Philosophy

Estimating depth, pose, and intrinsics separately compounds errors. A *pointmap* instead regresses, for every pixel, a dense 3D point expressed **in the coordinate frame of the other view** — from which camera pose, intrinsics, and depth all derive as projections. Multi-view geometry becomes a single regression problem.

## Functionality

- Siamese ViT encoder + alternating self/cross-attention decoder → per-pixel 3D points X^(1,2) per image pair.
- Per-pixel confidence head weights the regression loss (sky, textureless regions); MASt3R adds a local-feature descriptor head for matching.

## Used By

| Model | Role |
|-------|------|
| DUSt3R | Two-view pointmaps, symmetric heads on ViT-L |
| MASt3R | Pointmap + confidence + matching descriptors, coarse-to-fine reciprocal kNN |
| MoGe | Affine-invariant single-image pointmap + foreground mask + scale/focal heads |
| VGGT | One of four heads (camera/depth/pointmap/track) over alternating local/global attention |

## Features

- **Geometry from regression** — pose-free, calibration-free 3D reconstruction.
- **Confidence-weighted** — ambiguous pixels don't dominate the loss.

## Evolution

- **Predecessor**: depth + pose pipelines (SfM/MVS).
- **Successor**: VGGT scales it to many views with global attention; vggt-world forecasts pointmap latents as world state.
