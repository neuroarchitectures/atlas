# StreamPETR

## Overview

StreamPETR is a multi-view 3D detector built on **object-centric temporal modeling**: instead of building BEV feature maps or cost volumes every frame, it keeps a **memory queue of object queries from past frames** and propagates them into the current frame, aligning them with **motion-aware layer normalization** that accounts for ego-motion and the relative time interval.

- **Year:** 2023
- **Authors:** Wang et al.
- **Source:** arXiv:2303.11926 — *Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection*
- **Category:** DL/3D Detection

## Key Characteristics

- **Object queries as temporal state** — past queries are stored in a memory queue and reused, so temporal information is carried by object hypotheses rather than by feature maps.
- **Motion-aware layer normalization** — the query features are conditioned on ego-motion and the time gap, which is how historical queries are aligned to the current frame.
- **No BEV features, no cost volume** — this is the efficiency claim: temporal fusion happens on a few hundred queries, not on a dense grid.
- **Streaming inference** — each frame is processed once with the queue; only a few past frames are needed for the benefit.
- **Query-based detection head** — PETR-style 3D position-aware queries decoded into boxes.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Wang_et_al._2023_2303.11926.md`](references/papers/Wang_et_al._2023_2303.11926.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DETR3D / PETR / BEVFormer → StreamPETR; siblings: SparseBEV, Sparse4D, StreamPETR-v2; downstream: end-to-end autonomous-driving perception stacks).
