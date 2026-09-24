# Video Depth Anything

## Overview

Video Depth Anything extends the Depth Anything image model to video with the goal of **temporally consistent depth for arbitrarily long videos**. A lightweight temporal head is added on top of the spatial encoder, and a **temporal consistency loss** constrains the temporal depth gradient — so flicker and drift are suppressed without retraining the whole model.

- **Year:** 2025
- **Authors:** Chen et al. (TikTok / HKU)
- **Source:** arXiv:2501.12375 — *Video Depth Anything: Consistent Depth Estimation for Super-Long Videos*
- **Category:** DL/Depth

## Key Characteristics

- **Temporal head on a frozen/partly-finetuned spatial encoder** — reuses Depth Anything's strong single-frame depth rather than learning video depth from scratch.
- **Temporal consistency loss** constraining the temporal depth gradient, which is the direct fix for frame-to-frame flicker.
- Designed for **super-long videos**: inference is organized over segments/key frames so that a sequence of arbitrary length can be processed with bounded memory and stable results.
- Reported to improve both **temporal stability** and **per-frame depth quality** compared with applying an image model per frame.
- Keeps the relative-depth nature of Depth Anything; metric variants need extra scale supervision.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Chen_et_al._2025_2501.12375.md`](references/papers/Chen_et_al._2025_2501.12375.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (Depth Anything V1/V2 → Video Depth Anything; siblings: DepthCrafter, NVDS, ChronoDepth; successors: online/streaming variants).
