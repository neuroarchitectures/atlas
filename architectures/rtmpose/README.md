# RTMPose

## Overview

RTMPose is a **real-time multi-person 2D pose estimation** framework built on MMPose. It is the result of an empirical study of paradigm, backbone, training strategy, and deployment — and the winning combination is a **CSPNeXt backbone** with a **SimCC-style classification head** using **Unit Distance Scaling (UDP)**, which removes the expensive heatmap upsampling that dominates latency in keypoint models.

- **Year:** 2023
- **Authors:** Jiang et al. (Shanghai AI Laboratory)
- **Source:** arXiv:2303.07399 — *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose*
- **Category:** DL/Pose

## Key Characteristics

- **CSPNeXt backbone** — a CSP-style convnet chosen for throughput/GFlops efficiency on real hardware rather than for parameter count alone.
- **SimCC head** — pose estimation as **two 1D classification problems** (x and y coordinate bins) instead of 2D heatmap regression; removes deconvolution/upsampling layers and their latency.
- **UDP (Unit Distance Scaling)** — resolves the sub-pixel ambiguity of bin classification and fixes the flip-test inconsistency of naive SimCC.
- **Top-down paradigm**, and the paper studies the full stack (training strategy, deployment backends) rather than only the network.
- Reported: RTMPose-m achieves 75.8% AP on COCO at 90+ FPS on an Intel i7-11700 **CPU** and 430+ FPS on a single GPU — i.e. real-time without a datacenter GPU.
- Model family from tiny (-t) to large (-l), so latency and accuracy can be traded for the target device.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Jiang_et_al._2023_2303.07399.md`](references/papers/Jiang_et_al._2023_2303.07399.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (Hourglass / HRNet / HigherHRNet / SimCC → RTMPose; siblings: YOLO-Pose, ViTPose, DWPose, WiLoR).
