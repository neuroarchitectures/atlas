# Joint Space-Time Track Attention

## Design Philosophy

Factorized video attention (spatial then temporal) serializes information flow. For point tracking, treat the (time × tracks) grid as one token set and let attention mix across *both* axes simultaneously — so a point occluded in frame t can be inferred from where other tracks moved.

## Functionality

- Tokens = all query points + support points × all frames in the window; 8 transformer layers over the joint grid.
- Supports cross-track communication (points influence each other) and cross-frame reasoning in one operation.

## Used By

| Model | Role |
|-------|------|
| CoTracker3 | Joint (time × track) attention; windows chained with overlap for long videos |

## Features

- **Cross-track evidence sharing** — the key accuracy source over per-track models.
- **Parallel over all tracks** — tracks an arbitrary number of points jointly.

## Evolution

- **Predecessor**: TAPIR's two-stage-point-tracking (factorized).
- **Related**: divided-space-time-attention (TimeSformer) — the factorized counterpart for video classification.
