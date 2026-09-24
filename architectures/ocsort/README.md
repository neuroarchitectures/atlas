# OC-SORT

## Overview

SORT is *estimation-centric*: when a track is untracked, the Kalman filter keeps predicting from its own state, and error accumulates fast — so re-association often fails even when the object reappears, and the velocity direction becomes too noisy to use. OC-SORT makes the tracker **observation-centric** with two additions: **ORU** (observation-centric re-update, which heals accumulated error using virtual observations) and **OCM** (observation-centric momentum, which adds direction consistency to the association cost).

- **Year:** 2022
- **Authors:** Cao et al.
- **Source:** arXiv:2203.14360 — *Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking*
- **Category:** DL/Tracking

## Key Characteristics

- **ORU (observation-centric re-update)** — a third stage after predict/update: when a lost track is re-activated, virtual observations generated between the last-seen and the re-activating observation correct the accumulated error.
- **OCM (observation-centric momentum)** — direction consistency of tracks enters the association cost matrix, computed from observations rather than from noisy filter velocity.
- **Diagnosis first** — the paper identifies three SORT limitations: sensitivity to estimation noise, error accumulation over time, and being estimation-centric rather than observation-centric.
- **Still simple and online** — no appearance features, no batch processing; real-time.
- **Robustness target** — the gains concentrate on occlusion and non-linear motion, which is where SORT fails.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Cao_et_al._2022_2203.14360.md`](references/papers/Cao_et_al._2022_2203.14360.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (SORT / DeepSORT → OC-SORT; siblings: ByteTrack (low-score recovery), BoT-SORT; successors: Deep OC-SORT (adds appearance), hybrid trackers).
