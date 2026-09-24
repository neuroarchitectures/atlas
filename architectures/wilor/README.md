# WiLoR

## Overview

WiLoR is an end-to-end method for **3D hand localization and reconstruction in the wild**: a single-shot detector finds hands and regresses MANO parameters directly, then a **ViT-based refinement module** improves the mesh. It is trained on large in-the-wild data and handles occlusion and hand-object interaction without a separate detection stage.

- **Year:** 2024
- **Authors:** Potamias et al.
- **Source:** arXiv:2409.12259 — *WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild*
- **Category:** DL/3D Human Pose

## Key Characteristics

- **Single-shot** — detection, localization and mesh recovery in one forward pass; no cascade of detector → crop → regressor.
- **MANO parameter regression** — predicts pose, shape and camera parameters of the parametric hand model, so the output is a full 3D mesh.
- **ViT refinement** — a transformer refinement module that samples features around the initial estimate and improves it, which is what makes occluded and truncated hands work.
- **In-the-wild robustness** — trained with datasets spanning viewpoints, resolution and occlusion; generalizes beyond studio benchmarks.
- **Both hands** — handles hand-object interaction and two-hand cases rather than a single dominant hand.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Potamias_et_al._2024_2409.12259.md`](references/papers/Potamias_et_al._2024_2409.12259.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MANO → FrankMocap / METRO / HaMeR → WiLoR; siblings: Hand4Whole, Mesh Graphormer, InterHand; downstream: embodied/robotics hand pipelines).
