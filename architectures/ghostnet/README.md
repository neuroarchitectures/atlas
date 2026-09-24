# GhostNet

## Overview

Successful CNNs contain substantial redundancy in their feature maps — a characteristic that is widely known but had rarely been investigated in *neural architecture design*. GhostNet turns that redundancy into the design principle: instead of computing every feature map with an expensive convolution, compute a small set of intrinsic feature maps and derive the rest ("ghosts") with cheap linear transformations.

- **Year:** 2019
- **Authors:** Han et al. (Huawei Noah's Ark Lab, Peking University, University of Sydney)
- **Source:** arXiv:1911.11907 — *GhostNet: More Features from Cheap Operations*
- **Category:** DL/Efficient CNN

## Key Characteristics

- **Ghost module** — based on a set of intrinsic feature maps, applies a series of **cheap linear transformations** to generate many ghost feature maps that fully reveal the information underlying the intrinsic features.
- **Explicit exploitation of feature-map redundancy** — the paper states this redundancy is an important characteristic of successful CNNs but had rarely been investigated in architecture design.
- **Plug-and-play** — the Ghost module can be dropped in to upgrade existing convolutional neural networks, not just used inside GhostNet.
- **Ghost bottlenecks** — stacked Ghost modules form the bottlenecks that build GhostNet.
- **Beats MobileNetV3 at similar cost** — **75.7% top-1** on ImageNet ILSVRC-2012 with comparable computational cost.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Han_et_al._2019_1911.11907.md`](references/papers/Han_et_al._2019_1911.11907.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MobileNetV1/V2/V3 → GhostNet; siblings: ShuffleNet V2, MobileNetV4, StarNet; contrast: pruning / quantization / distillation, which are upper-bounded by their pre-trained baselines).
