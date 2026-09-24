# TAPIR

## Overview

Tracking-any-point has two competing designs: TAP-Net matches **globally per frame**, which is robust to occlusion but jittery because it ignores temporal continuity; PIPs **refines locally over time**, which is smooth but breaks under occlusion and is slow. TAPIR observes they are complementary and combines them: per-frame global matching for initialization, then temporal refinement for a realistic trajectory.

- **Year:** 2023
- **Authors:** Doersch et al. (DeepMind)
- **Source:** arXiv:2306.08637 — *TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement*
- **Category:** DL/Tracking

## Key Characteristics

- **Two-stage complementarity** — global per-frame matching (occlusion-robust, coarse) + temporal local refinement (smooth, precise); this pairing *is* the contribution.
- **Per-frame initialization** — every frame produces an independent match, so the method does not depend on a sequential chain that can derail.
- **Temporal refinement** — a local search over the neighbourhood smooths the track across time.
- **Occlusion and uncertainty prediction** — the model outputs whether a point is visible, which matters for any downstream use of trajectories.
- **Large benchmark gains** — ~20% better than TAP-Net on TAP-Vid-DAVIS and ~20% better than PIPs on TAP-Vid-Kinetics, with much faster inference than PIPs.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Doersch_et_al._2023_2306.08637.md`](references/papers/Doersch_et_al._2023_2306.08637.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (optical flow → TAP-Net / PIPs → TAPIR; siblings: CoTracker (also in catalog as `cotracker3`), BootsTAP; contrast: box-level MOT (ByteTrack, OC-SORT)).
