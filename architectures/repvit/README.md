# RepViT

## Overview

Lightweight ViTs had started beating lightweight CNNs on both accuracy and latency on mobile devices. Rather than adopt ViTs wholesale, RepViT asks what a *CNN* would look like if it borrowed the design choices that made those ViTs good — then shows the result beats the lightweight ViTs on their own terms.

- **Year:** 2023
- **Authors:** Wang et al. (Tsinghua, MEITUAN, Auckland University of Technology)
- **Source:** arXiv:2307.09283 — *RepViT: Revisiting Mobile CNN From ViT Perspective*
- **Category:** DL/Efficient CNN

## Key Characteristics

- **Re-examine the disparities** — prior work noted structural connections between lightweight ViTs and CNNs, but the differences in **block structure, macro design and micro design** had not been adequately examined; that is RepViT's subject.
- **Incremental enhancement of MobileNetV3** — ViT architectural designs are integrated step by step, ending at a family of **pure lightweight CNNs**.
- **Beats lightweight ViTs** — over **80% top-1 on ImageNet at 1.0 ms latency on an iPhone 12**, described as the first time for a lightweight model.
- **RepViT-SAM** — nearly **10× faster** inference than the advanced MobileSAM when RepViT is used as the SAM backbone.
- **Favorable latency across vision tasks**, not just classification.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Wang_et_al._2023_2307.09283.md`](references/papers/Wang_et_al._2023_2307.09283.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MobileNetV3, RepVGG → RepViT; siblings: MobileNetV4, StarNet, FasterNet, MobileSAM; contrast: lightweight ViTs the paper benchmarks against).
