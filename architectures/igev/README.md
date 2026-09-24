# IGEV-Stereo

## Overview

RAFT showed all-pairs correlation works for stereo, but all-pairs correlations carry **no non-local geometry knowledge** and struggle with local ambiguities in ill-posed regions (textureless surfaces, occlusions). IGEV-Stereo builds a **geometry encoding volume** that encodes geometry, context and local matching details together — and crucially uses it to regress an accurate starting disparity, so the ConvGRU iterations converge fast.

- **Year:** 2023
- **Authors:** Xu et al. (Samsung Research China - Beijing, University of Adelaide, East China Normal University)
- **Source:** arXiv:2303.06615 — *Iterative Geometry Encoding Volume for Stereo Matching*
- **Category:** DL/Stereo Depth

## Key Characteristics

- **Geometry encoding volume (GEV)** — a combined volume encoding geometry and context information as well as local matching details; it is indexed iteratively to update the disparity field.
- **Regressed starting point** — GEV is used to regress an accurate initial disparity instead of starting from zero, which speeds convergence of the ConvGRU iterations.
- **Non-local geometry knowledge** — the explicit answer to what all-pairs correlation lacks in ill-posed regions.
- **Ranked 1st on KITTI 2015 and KITTI 2012 (Reflective)** among published methods, and the fastest among the top ten; strong cross-dataset generalization.
- **Extends to MVS** — the same GEV idea is applied to multi-view stereo.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Xu_et_al._2023_2303.06615.md`](references/papers/Xu_et_al._2023_2303.06615.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (PSMNet / GA-Net → RAFT-Stereo → IGEV-Stereo; siblings: CREStereo, FoundationStereo; contrast: RAFT-Stereo relies on local correlation lookup without a geometry encoding volume).
