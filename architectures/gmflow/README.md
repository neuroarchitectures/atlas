# GMFlow

## Overview

Optical flow pipelines (PWC-Net coarse-to-fine, RAFT iterative refinement) regress flow from a *local* cost volume, so they need many refinement stages to handle large displacements — and that sequential chain is exactly what makes them slow. GMFlow reformulates flow as a **global matching** problem solved by a transformer, resolving large displacement in **one refinement**.

- **Year:** 2021
- **Authors:** Xu et al.
- **Source:** arXiv:2111.13680 — *GMFlow: Learning Optical Flow via Global Matching*
- **Category:** DL/Optical Flow

## Key Characteristics

- **Global matching instead of local regression** — a transformer matches features across the whole image rather than regressing from a local window, which is what removes the multi-stage requirement.
- **One refinement, not many** — the paper reports outperforming 31-refinement RAFT on Sintel with a single refinement and lower runtime.
- **Global correlation** — all-pairs similarity computed over features; the transformer then reasons over it.
- **Flow propagation** — matching predictions are propagated into a dense flow field, with a self-supervised refinement stage that fixes unmatched regions.
- **Speed-accuracy rethink** — the argument is structural: sequential refinements make latency hard to optimize, so reducing their number is worth more than tuning them.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Xu_et_al._2021_2111.13680.md`](references/papers/Xu_et_al._2021_2111.13680.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (PWC-Net → RAFT → GMFlow; siblings: FlowFormer (cost-memory transformer), SEA-RAFT, DICL, GLU-Net; successor: GMFlow+ / unified matching pipelines).
