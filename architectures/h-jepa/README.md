# H-JEPA

## Overview

H-JEPA (Hierarchical JEPA) solves a fundamental limitation of single-timescale JEPA world models for long-range planning: prediction errors accumulate over multi-step rollouts, and a single latent space must simultaneously support low-level dynamics (fast) and high-level goal matching (slow). H-JEPA trains a **hierarchy of JEPA layers**, each operating in its own latent space at a progressively coarser temporal stride. Each layer predicts further into the future than the layer below. Planning is **top-down**: the top layer's prediction becomes the subgoal for the layer below, which plans toward it, and so on down to actions. On Visual AntMaze, a three-layer hierarchy lifts success rate from 18% to 73% using less planning compute than the flat baseline.

- **Year:** 2026
- **Authors:** Chenlin Zhang, Mike Rabbat, Yann LeCun, Randall Balestriero (Meta FAIR / Mila)
- **Source:** arXiv:2610.06805 — *H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning*
- **Category:** DL/World-Model

## Key Characteristics

- **Hierarchical temporal abstraction**: each layer operates at a coarser time step (Δt), predicting further into the future.
- **Separate latent spaces per layer**: each level has its own representation, preventing the single-latent-space bottleneck.
- **Top-down subgoal planning**: the top layer predicts a distant-future subgoal; each lower layer plans toward its upper layer's subgoal; the bottom layer outputs actions.
- **Upper layers discard fast unpredictable details**: by design, higher levels retain only slow, task-relevant state, making long-range prediction tractable.
- **Visual AntMaze: 18% → 73%** success rate with three-layer hierarchy, using less planning compute.
- **Validated on DROID real robot video**: demonstrates applicability beyond simulated environments.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhang_et_al._2026_2610.06805.md`](references/papers/Zhang_et_al._2026_2610.06805.md)

## Related Architectures

See `architecture.md` → Evolution section (V-JEPA 2-AC → H-JEPA; sibling: LeHami provides stable single-level dynamics; H-JEPA provides temporal abstraction across levels).
