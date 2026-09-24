# Scale-and-Shift-Invariant Depth Loss

## Design Philosophy

Monocular depth is affine-ambiguous: without knowing the true scale and shift, comparing predicted and GT depth pixel-wise punishes valid predictions. Align prediction to target with the *optimal scale and shift* before the loss (plus a gradient term for sharp boundaries) — supervision becomes affine-invariant.

## Functionality

- Compute s*, t* minimizing ‖s·ẑ + t − z‖²; loss on aligned residuals; gradient-matching term on spatial derivatives.
- Metric3D's Random Proposal Normalization applies the same invariant on randomly cropped patches instead of whole images, preserving local geometry.

## Used By

| Model | Role |
|-------|------|
| Depth Anything V2 | Affine-invariant relative depth training |
| Metric3D | Random-proposal normalization variant around the canonical transform |

## Features

- **Ordering-focused supervision** — the model learns relative geometry, scale handled separately.
- **Patch-level variant** adds local-structure pressure.

## Evolution

- **Predecessor**: MiDaS scale-shift alignment (2020), eigen-loss.
- **Related**: canonical-camera-transform — handle ambiguity by warping instead of by loss; per-region-optimal-supervision (finer-grained alignment regions).
