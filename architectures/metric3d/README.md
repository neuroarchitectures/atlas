# Metric3D

## Overview

Monocular depth split into two camps: affine-invariant models (MiDaS, LeReS) that generalize but recover no scale, and metric models that are accurate but overfit one camera setup. Metric3D attacks the **camera** side of that trade-off: transform everything into a **canonical camera space**, train there, then undo the transformation to get metric scale — which makes it possible to train on 8 million images from 11 datasets with tens of thousands of different cameras.

- **Year:** 2023
- **Authors:** Yin et al. (Intel Labs, CUHK, The University of Adelaide, ShanghaiTech)
- **Source:** arXiv:2307.10984 — *Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image*
- **Category:** DL/Monocular Depth

## Key Characteristics

- **Canonical camera space** — all training data is transformed to one canonical camera, removing camera-model variance from the learning problem; a **de-canonical transformation** restores metric information at inference.
- **Two transformation routes** — either adjust image appearance to simulate the canonical camera, or transform the ground-truth labels for supervision; camera models are not encoded into the network, so the method drops into existing architectures.
- **Random proposal normalization loss** — the scale-shift invariant loss applied to *randomly cropped patches* rather than the whole image, because whole-image normalization squeezes fine-grained depth differences.
- **Scale as the enabler** — 8M images from 11 datasets with tens of thousands of cameras, which is what zero-shot transfer needs.
- **Downstream-usable metric output** — predicted metric depths reduce scale drift in monocular SLAM and enable large-scale 3D reconstruction; won the 2nd Monocular Depth Estimation Challenge.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Yin_et_al._2023_2307.10984.md`](references/papers/Yin_et_al._2023_2307.10984.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MiDaS / LeReS affine-invariant → Metric3D; siblings: UniDepth (predicts the camera), Depth Pro (estimates focal length), Depth Anything (relative); successors: Metric3D v2, Depth Anything with metric heads).
