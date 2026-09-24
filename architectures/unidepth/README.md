# UniDepth

## Overview

Monocular metric depth methods either generalize (affine-invariant models like MiDaS) or output metric scale (domain-specific supervised models) — but not both. UniDepth is built to be **universal**: it predicts a metric 3D point per pixel from a single image **without being given camera intrinsics**, by inferring the camera as part of the task and disentangling camera from depth by construction.

- **Year:** 2024
- **Authors:** Piccinelli et al. (ETH Zurich, University of Trento)
- **Source:** arXiv:2403.18913 — *UniDepth: Universal Monocular Metric Depth Estimation*
- **Category:** DL/Monocular Depth

## Key Characteristics

- **Camera is predicted, not provided** — a **self-prompting camera module** learns a dense non-parametric camera representation, so no intrinsics metadata is required at inference.
- **Pseudo-spherical output space** — the output space is parameterized as azimuth, elevation and log-depth, making camera and depth components orthogonal by design instead of entangled as in Cartesian coordinates.
- **Geometric invariance loss** — camera-conditioned depth features from two views of the same image must be reciprocally consistent, which further decouples camera from scene geometry.
- **Spherical harmonic camera encoding** — camera components are embedded with Laplace spherical harmonic encoding.
- **Benchmarking as a contribution** — the paper re-evaluates seven prior state-of-the-art methods on ten datasets under a fair zero-shot protocol, and UniDepth still ranks first on the official KITTI Depth Prediction Benchmark.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Piccinelli_et_al._2024_2403.18913.md`](references/papers/Piccinelli_et_al._2024_2403.18913.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MiDaS / LeReS affine-invariant → metric supervised → UniDepth; siblings: Depth Pro, Metric3D, Depth Anything; successors: UniDepthV2, follow-up universal MMDE work).
