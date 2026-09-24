# SEA-RAFT

## Overview

SEA-RAFT keeps RAFT's recurrent refinement structure and changes three things: a **mixture-of-Laplace loss** instead of the standard L1, **direct regression of an initial flow** rather than starting from zero, and **rigid-motion pretraining** for generalization. The result is state-of-the-art accuracy on Spring (3.686 EPE, 0.363 1px) at **≥2.3× the speed** of comparable methods.

- **Year:** 2024
- **Authors:** Wang et al.
- **Source:** arXiv:2405.14793 — *SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow*
- **Category:** DL/Optical Flow

## Key Characteristics

- **Mixture-of-Laplace loss** — replaces the usual L1 flow loss, better modelling the heavy-tailed distribution of flow errors.
- **Direct initial flow regression** — the model regresses an initial flow instead of starting at zero, so iterative refinement converges in fewer steps; this is the main efficiency lever.
- **Rigid-motion pretraining** — synthetic rigid-motion pretraining improves cross-dataset generalization (best reported generalization on KITTI and Spring).
- **Keeps RAFT's structure** — cost volume + recurrent GRU refinement + convex upsampling; the changes are in the objective and initialization, not the topology.
- **Pareto frontier** — best accuracy-efficiency trade-off reported across Spring, Sintel and KITTI; the smallest model runs at 21 fps.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Wang_et_al._2024_2405.14793.md`](references/papers/Wang_et_al._2024_2405.14793.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (RAFT → SEA-RAFT; siblings: FlowFormer, GMFlow, DIP; contrast: FlowFormer changes the topology, SEA-RAFT does not).
