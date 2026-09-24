# Object-Query Memory Queue

## Design Philosophy

Temporal 3D detection usually keeps a dense BEV grid and warps it. StreamPETR's alternative: keep a *fixed-length FIFO of past object queries* (with timestamps) as the temporal state — the objects themselves are the memory. New queries attend to the queue; after decoding, the new queries are pushed back.

## Functionality

- Queue of ~N past frames' top object queries + normalized time intervals.
- Motion-aware LayerNorm conditions queued queries on the ego-motion transform (rotation + translation between frames) — aligning old queries geometrically without resampling.
- Current queries cross-attend to the aligned queue; decoded queries re-enter the queue.

## Used By

| Model | Role |
|-------|------|
| StreamPETR | Streaming multi-view 3D detection; no BEV grid, no feature warping |

## Features

- **Sparse temporal memory** — memory size is the query budget, not the map size.
- **Motion-aware normalization** — geometry injected at the conditioning level, not via explicit warping.

## Evolution

- **Predecessor**: Sparse4D, PETRv2 (feature-level temporal fusion).
- **Related**: temporal-bev-attention (dense-grid recurrence); streaming-memory-bank (SAM 2) — the same "bounded temporal memory" idea in segmentation.
