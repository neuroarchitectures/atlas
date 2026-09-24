# RAFT-Stereo

## Overview

Deep stereo pipelines build a 4D cost volume and process it with expensive 3D convolutions, which are slow and generalize poorly outside the training domain. RAFT-Stereo replaces the 3D conv stack with **multi-level recurrent field transforms**: GRU update operators read from a correlation (cost) volume pyramid and iteratively refine disparity, with convex upsampling to full resolution.

- **Year:** 2021
- **Authors:** Lipson et al.
- **Source:** arXiv:2109.07547 — *RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching*
- **Category:** DL/Stereo Depth

## Key Characteristics

- **No 3D convolutions** — a recurrent GRU operator over the cost volume replaces the 3D conv hourglass, cutting cost and raising the operating resolution.
- **Multi-level cost volume** — correlation volumes at several resolutions with cross-connections, so large and small disparities are handled together.
- **Iterative disparity refinement** — the GRU updates disparity over iterations, the stereo analogue of RAFT's flow refinement.
- **Convex upsampling** — disparity is refined at reduced resolution and upsampled learnably to full resolution only at the end, which is why megapixel images fit in memory.
- **Zero-shot cross-dataset generalization** — the headline practical property: transfers to ETH3D, KITTI, Middlebury without retraining, unlike 3D-conv stereo networks.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Lipson_et_al._2021_2109.07547.md`](references/papers/Lipson_et_al._2021_2109.07547.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DispNet / GC-Net / PSMNet / GA-Net → RAFT-Stereo; siblings: HITNet, IGEV, CREStereo; successors: FoundationStereo (zero-shot foundation stereo), Fast-FoundationStereo).
