# Deformable Convolution (DCN-v2)

## Design Philosophy

A convolution where the **sampling grid is learnable and input-dependent**. The philosophy: a standard conv samples a fixed grid (e.g., 3×3 around each pixel); a deformable conv learns 2D offsets for each sample point, so the kernel can adapt to geometric transformations (rotation, scale, non-rigid deformation). DCN-v2 adds a modulation scalar per sample point, making the offsets multiplicative.

## Functionality

- **Offset learning**: A small conv predicts 2D offsets `(Δx, Δy)` for each of the `k×k` sample points, per kernel position.
- **Modulation (v2)**: An additional scalar `m_i ∈ [0,1]` per sample point, learned the same way, modulates each sample's contribution.
- **Deformable conv**: Sample the input at `(x + Δx, y + Δy)` via bilinear interpolation, weight by `m_i`, apply the conv kernel.
- **Regular conv as special case**: Set offsets to 0, modulations to 1 → standard conv.

## Used By

| Model | Role |
|-------|------|
| DCN / DCN-v2 | The defining operation |
| Deformable DETR | Deformable attention for detection |
- Used where geometric adaptation matters (detection, segmentation).

## Features

- **Geometric adaptation**: The kernel shape adapts to the input's geometry.
- **Modulation (v2)**: Per-sample-point weighting — better than v1's unweighted sampling.
- **Drop-in**: Same I/O shape as a standard conv, just with a learned offset field.

## Evolution

- **Predecessor**: Spatial Transformer Networks (global affine); dilated/atrous convs (fixed shape).
- **Successor**: Deformable attention (DETR); the pattern of "learnable sampling" influenced attention variants.
