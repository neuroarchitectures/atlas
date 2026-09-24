# HRNet

## Overview

Classification backbones (ResNet, VGG) reduce resolution in series and, for position-sensitive tasks, recover high resolution afterwards with upsampling or dilated convolutions. HRNet inverts this: it **keeps a high-resolution representation throughout** by running high-to-low resolution convolution streams **in parallel** and repeatedly exchanging information between them.

- **Year:** 2019
- **Authors:** Wang et al.
- **Source:** arXiv:1908.07919 — *Deep High-Resolution Representation Learning for Visual Recognition*
- **Category:** DL/Backbone

## Key Characteristics

- **Parallel multi-resolution streams** — instead of a serial high→low chain, streams at different resolutions run side by side from stage 2 onward.
- **Repeated multi-scale fusion** — information is exchanged across resolutions at every stage, not only at the end, so low-resolution semantics continuously enrich high-resolution detail.
- **High resolution maintained end to end** — no lossy encode-then-decode recovery step; the output representation is both semantically rich and spatially precise.
- **General backbone** — applied to pose estimation, semantic segmentation, and object detection, not a single task.
- **The pose-estimation default** — HRNet-W32/W48 became the standard backbone for top-down keypoint models.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Wang_et_al._2019_1908.07919.md`](references/papers/Wang_et_al._2019_1908.07919.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (ResNet / FPN / Hourglass / U-Net → HRNet; siblings: HRNet-OCR, HigherHRNet, Lite-HRNet; successors: ViTPose (transformer backbone), RTMPose).
