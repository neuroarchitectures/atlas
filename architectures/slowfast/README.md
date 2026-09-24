# SlowFast Networks

## Overview

In image recognition it is customary to treat the two spatial dimensions symmetrically, justified by natural images being roughly isotropic and shift-invariant. **Video is different**: motion is the spatiotemporal counterpart of orientation, but **not all spatiotemporal orientations are equally likely** — slow motions are more likely than fast ones. SlowFast argues there is therefore no reason to treat space and time symmetrically, and **factors** the architecture instead.

- **Year:** 2018
- **Authors:** Feichtenhofer et al. (Facebook AI Research (FAIR))
- **Source:** arXiv:1812.03982 — *SlowFast Networks for Video Recognition*
- **Category:** DL/Video Recognition

## Key Characteristics

1. **Slow pathway** — operates at **low frame rate** to capture **spatial semantics**; categorical semantics evolve slowly, so they can be refreshed relatively slowly.
2. **Fast pathway** — operates at **high frame rate** to capture **motion at fine temporal resolution**; the motion being performed evolves much faster than subject identity.
3. **Fast pathway is very lightweight** — it can be made so by **reducing its channel capacity**, yet still learns useful temporal information for video recognition.
4. **Strong results on action classification and detection in video** — with large improvements pin-pointed as contributions by the SlowFast concept.
5. **State-of-the-art on major benchmarks** — Kinetics, Charades and AVA.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Feichtenhofer_et_al._2018_1812.03982.md`](references/papers/Feichtenhofer_et_al._2018_1812.03982.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (two-stream networks, spatiotemporal 3D CNNs → SlowFast; siblings: TimeSformer, ViViT, UniFormer, VideoMAE; contrast: approaches based on spatiotemporal convolutions, which treat space and time symmetrically).
