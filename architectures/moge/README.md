# MoGe

## Overview

MoGe estimates **open-domain monocular geometry**: from a single image it predicts an **affine-invariant 3D point map**, from which depth, camera focal length, and foreground geometry follow. Its central insight is about supervision — most datasets have reliable geometry only for *parts* of an image, so training must use **optimal supervision** per region rather than a single global scale/shift alignment.

- **Year:** 2024
- **Authors:** Wang et al. (USTC / Microsoft Research / Harvard / Tsinghua University)
- **Source:** arXiv:2410.19115 — *MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision*
- **Category:** DL/3D-Reconstruction

## Key Characteristics

- **Affine-invariant point map** output: a dense 3D point per pixel plus a **global scale/shift** that can be estimated, giving metric-consistent geometry instead of only ordinal depth.
- **Optimal training supervision**: per-region scale/shift alignment chooses the best supervision available in each part of a training sample, so incomplete ground truth (sky, far field, missing annotations) does not corrupt training.
- **Foreground mask head** — separates the reconstructable foreground from ambiguous background regions.
- **Camera focal length** prediction, which is what turns the point map into usable geometry.
- Generalizes across open-domain images (indoor, outdoor, object-centric, generated images).
- Family: MoGe and MoGe-2 (metric-scale variants), with ViT-B/L encoders used in production pipelines.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Wang_et_al._2024_2410.19115.md`](references/papers/Wang_et_al._2024_2410.19115.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (Depth Anything / UniDepth → MoGe; siblings: DUSt3R, MASt3R, Metric3D, GeoCalib).
