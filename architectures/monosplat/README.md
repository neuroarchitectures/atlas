# MonoSplat

## Overview

Generalizable 3D Gaussian reconstruction approaches often break down on **unfamiliar visual content and layouts**, because their understanding of the visual world is **constrained by the training data distribution** — a zero-shot domain generalization failure. MonoSplat's premise: contemporary **monocular depth foundation models** (MiDaS, Depth Anything) are trained on extensive datasets and predict monocular depth across diverse visual domains exceptionally well, so if broad visual understanding is what makes Gaussian reconstruction generalize, a pretrained monocular depth estimator should be enough to build a widely applicable framework.

- **Year:** 2025
- **Authors:** Liu et al. (Nanyang Technological University, University of Adelaide, Shandong University, Westlake University)
- **Source:** arXiv:2505.15185 — *MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models*
- **Category:** DL/Neural Scene Representation

## Key Characteristics

1. **Converts a pretrained depth model into a Gaussian reconstruction model** — a simple yet effective framework; the depth foundation model stays **frozen**.
2. **Mono-Multi Feature Adapter** — transforms **monocular features from the frozen depth foundation model into multi-view features with cross-view awareness**; this is where monocular priors become multi-view consistent.
3. **Integrated Gaussian Prediction module** — synergistically integrates monocular and multi-view features to generate **precise Gaussian primitives**.
4. **Remarkable efficiency with minimal trainable parameters** — a consequence of the frozen-backbone design.
5. **Unprecedented zero-shot generalization** — the novel integration of monocular depth priors establishes new performance standards across diverse real-world scenarios, which is the problem the paper set out to solve.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Liu_et_al._2025_2505.15185.md`](references/papers/Liu_et_al._2025_2505.15185.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (NeRF, pixelSplat, MVSplat, FreeSplat/eFreeSplat, monocular depth foundation models → MonoSplat; siblings: dust3r; contrast: multi-view-trained generalizable methods, whose understanding is bounded by the training distribution).
