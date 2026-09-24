# MASt3R

## Overview

MASt3R (Matching And Stereo 3D Reconstruction) extends **DUSt3R** with a second output: a dense **local feature map** per view. The 3D regression branch keeps DUSt3R's pointmap accuracy while the descriptor branch enables **fast reciprocal matching**, replacing expensive global-optimization matching in pose estimation and visual localization.

- **Year:** 2024
- **Authors:** Leroy et al. (Naver Labs Europe / ENS)
- **Source:** arXiv:2406.09756 — *Grounding Image Matching in 3D with MASt3R*
- **Category:** DL/3D-Reconstruction

## Key Characteristics

- Two heads over one shared siamese ViT-L encoder + cross-attention decoder: **pointmap head** (dense 3D) and **local feature head** (pixel-wise descriptors).
- **Fast reciprocal matching**: coarse-to-fine kNN matching with a reciprocal criterion over the dense descriptors — reported to be almost two orders of magnitude faster than the optimization-based matching used with DUSt3R, while improving pose accuracy.
- Makes the model usable for **visual localization** — match query image features against a database, then recover pose from the matched 3D points.
- Retains DUSt3R's property of requiring **no camera calibration and no pose prior**.
- The descriptor branch is the architectural differentiator: DUSt3R alone cannot match reliably across large baselines or between unrelated pairs.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Leroy_et_al._2024_2406.09756.md`](references/papers/Leroy_et_al._2024_2406.09756.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DUSt3R → MASt3R → MASt3R-SfM / Spann3R, Fast3R, VGGT).
