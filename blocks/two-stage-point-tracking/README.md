# Two-Stage Point Tracking

## Design Philosophy

Pure local tracking drifts; pure global matching is noisy per-frame. Factorize: stage 1 performs *global matching* against the current frame (robust re-initialization, occlusion recovery); stage 2 *refines locally* over a temporal window. An explicit visibility head decides when to trust each stage.

## Functionality

- Stage 1: attention-based global match of each tracked point against all frame features.
- Stage 2: windowed temporal attention refining positions across neighboring frames; visibility/uncertainty score per point per frame.

## Used By

| Model | Role |
|-------|------|
| TAPIR | Per-frame global matching + local refinement; also the engine of CoTracker's family |

## Features

- **Occlusion-robust re-anchoring** — global stage recovers what local refinement loses.
- **Explicit visibility** — downstream consumers can weight by reliability.

## Evolution

- **Predecessor**: PIPS (patch tracking with GRU memory).
- **Successor**: CoTracker3's joint-spacetime-attention collapses the two stages into one token grid.
