# VGGT-World

## Overview

VGGT-World is a **geometry world model**: instead of forecasting future *video frames* (which spends most capacity on photometric detail and stays geometrically inconsistent), it forecasts the temporal evolution of **geometry features**. It re-purposes the latent tokens of a **frozen VGGT** as the world state and trains a lightweight **temporal flow transformer** to predict their future trajectory autoregressively.

- **Year:** 2026
- **Authors:** Sun et al. (UQMM Lab, The University of Queensland / Beijing Jiaotong University)
- **Source:** arXiv:2603.12655 — *VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model*
- **Category:** DL/World-Model

## Key Characteristics

- **No video generation** — the model predicts geometry latents, sidestepping the cost and inconsistency of pixel-space future prediction.
- **Frozen geometry foundation model (GFM) as the state encoder** — VGGT's tokens are the world state; only the temporal dynamics are learned, so the model inherits VGGT's 3D competence.
- **Lightweight temporal flow transformer** — autoregressive predictor over the latent token sequence; training uses a flow-matching-style objective rather than pixel reconstruction.
- Decouples *what the world looks like* (handled by the frozen encoder) from *how the world changes* (learned) — a much cheaper learning problem than full generative world models.
- Two technical challenges named by the authors: high-dimensional latent dynamics and autoregressive drift over long horizons.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Sun_et_al._2026_2603.12655.md`](references/papers/Sun_et_al._2026_2603.12655.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (V-JEPA / I-JEPA, Dreamer → VGGT-World; base: VGGT; siblings: action-conditioned geometry world models).
