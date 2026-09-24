# Depth Pro

## Overview

Zero-shot monocular depth models produce smooth, low-resolution maps — fine, except for applications like single-image novel view synthesis, where boundary quality is exactly what shows. Depth Pro targets **metric, high-resolution, sharp** depth at speed: a 2.25-megapixel depth map in 0.3 seconds, with absolute scale and **no camera intrinsics required**.

- **Year:** 2024
- **Authors:** Bochkovskii et al. (Apple)
- **Source:** arXiv:2410.02073 — *Depth Pro: Sharp Monocular Metric Depth in Less Than a Second*
- **Category:** DL/Monocular Depth

## Key Characteristics

- **Sharpness as a first-class target** — the paper notes that boundary accuracy was not being measured properly, so it introduces **dedicated evaluation metrics for boundary accuracy** in depth maps.
- **Multi-scale vision transformer** — an efficient multi-scale ViT for dense prediction, paired with a convolutional decoder for fine boundary tracing.
- **Real + synthetic training mixture** — a training protocol combining both to get metric accuracy *and* high-frequency detail.
- **Focal length from one image** — state-of-the-art focal length estimation from a single image, which is what allows metric output without supplied intrinsics.
- **Speed** — 2.25 MP depth map in 0.3 s on a standard GPU.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Bochkovskii_et_al._2024_2410.02073.md`](references/papers/Bochkovskii_et_al._2024_2410.02073.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MiDaS → UniDepth → Depth Pro; siblings: Metric3D, Depth Anything V2, MoGe-2; contrast: affine-invariant models recover no scale).
