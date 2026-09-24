# CoTracker3

## Overview

CoTracker3 is a **point tracker**: given a video and a set of query points, it produces dense per-frame trajectories with visibility flags. Its contribution is as much about data as architecture — a simple **pseudo-labelling protocol** that trains on unlabelled real videos, so a model pre-trained on synthetic data (Kubric) needs only a tiny fraction of the usual real data to exceed the state of the art.

- **Year:** 2024
- **Authors:** Karaev et al. (Meta AI / Visual Geometry Group, University of Oxford)
- **Source:** arXiv:2410.11831 — *CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos*
- **Category:** DL/Tracking

## Key Characteristics

- **Joint tracking**: all query points (plus learned support/context points) are tracked together by a transformer, so points help each other instead of being tracked independently.
- **Sliding-window inference** with iterative refinement of trajectories, which is what makes long videos tractable.
- **No optical flow prior** — unlike TAPIR, it does not need a flow model; the architecture is simpler.
- **Pseudo-labelling protocol**: a teacher tracker labels real videos, the student is trained on those labels; reported to outperform prior SoTA using only ~0.1% of the real training data used by competitors.
- Notably robust to occlusions; predicts an explicit **visibility** flag per point per frame.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Karaev_et_al._2024_2410.11831.md`](references/papers/Karaev_et_al._2024_2410.11831.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (RAFT-based trackers, TAPIR, PIPs → CoTracker → CoTracker3; related: BootsTAPIR, LocoTrack).
