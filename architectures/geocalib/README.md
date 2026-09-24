# GeoCalib

## Overview

GeoCalib recovers **camera parameters from a single image** — gravity direction (roll and pitch), focal length, and optionally distortion — by combining a network with a **differentiable geometric optimization** layer. The network predicts dense **perspective fields** (latitude and up-vector), and the camera model is then fitted to those fields by least squares inside the graph.

- **Year:** 2024
- **Authors:** Veicht et al. (ETH Zurich / Microsoft Mixed Reality & AI Lab)
- **Source:** arXiv:2409.06704 — *GeoCalib: Learning Single-image Calibration with Geometric Optimization*
- **Category:** DL/3D-Reconstruction

## Key Characteristics

- **Hybrid learning + optimization**: the network does perception (perspective fields), the optimization layer does geometry (camera fitting) — so the output is guaranteed to be a valid camera model rather than a raw regression.
- **Perspective field representation**: dense per-pixel **latitude** and **up-vector** predictions, which encode gravity direction and intrinsics in a form that is easy to fit.
- **Neural objective with uncertainty weighting** — the network learns which pixels to trust, and the fitter uses those weights in a weighted least-squares (Levenberg–Marquardt-style) solve.
- Differentiable end-to-end, so the network learns to produce fields that are *easy to calibrate from*, not just fields that look right.
- Works on in-the-wild images without calibration patterns; useful as a preprocessing step for SfM, SLAM, and metric 3D.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Veicht_et_al._2024_2409.06704.md`](references/papers/Veicht_et_al._2024_2409.06704.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (classical vanishing-point calibration → PerspectiveFields → GeoCalib; related: MoGe, DUSt3R, VGGT, Metric3D).
