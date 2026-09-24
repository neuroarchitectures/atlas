# All-Pairs Correlation Pyramid

## Design Philosophy

Correspondence between two frames is a matching problem — so compute it explicitly: the inner product of every pixel pair of two feature maps forms a 4D correlation volume, average-pooled into a multi-scale pyramid for cheap fixed-radius lookup at large displacements. The volume is computed once and reused by every refinement iteration.

## Functionality

- `C(u,v) = f1(u) · f2(v)` for all pixel pairs; 4 levels by 2× average pooling (radius-4 bilinear lookup per level).
- GMFlow variant: the global correlation feeds a transformer for one-shot global matching instead of iterative lookup.

## Used By

| Model | Role |
|-------|------|
| RAFT | 4-level pyramid, radius-4 lookup, computed once |
| RAFT-Stereo | Multi-level volume with cross-connections for disparity |
| SEA-RAFT | Correlation volume as GEV-free matching evidence |
| CoTracker3 | Local correlation sampling around current point estimates |
| GMFlow | Global correlation → transformer matching |

## Features

- **Explicit matching evidence** — beats implicit attention for correspondence.
- **Compute once, query many** — amortized over refinement iterations.

## Evolution

- **Predecessor**: FlowNet correlation layers (single scale, no pyramid).
- **Successor**: geometry-encoding-volume (IGEV) fuses geometry/context into the volume; direct-transformer matching (GMFlow) removes iteration.
