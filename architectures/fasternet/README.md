# FasterNet

## Overview

"Reduce FLOPs" has been the standing recipe for faster networks, but measured latency often refuses to follow — the paper's example: CycleMLP-B1 has half the FLOPs of ResNet50 yet runs *slower* (116.1 ms vs 73.0 ms). FasterNet's diagnosis is **FLOPS**, not FLOPs: depthwise and group convolutions spend their time on frequent memory access, so they compute few FLOPs per second on real hardware.

- **Year:** 2023
- **Authors:** Chen et al. (Northeast Petroleum University, Hong Kong Baptist University, Nanyang Technological University, University of Illinois Urbana-Champaign)
- **Source:** arXiv:2303.03667 — *Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks*
- **Category:** DL/Efficient CNN

## Key Characteristics

- **FLOPs ≠ latency** — the explicit framing: reducing FLOPs does not translate into the same reduction in latency, and can make it worse.
- **PConv (partial convolution)** — applies a regular convolution to only **part of the input channels**, leaving the rest untouched; exploits feature-map redundancy, cuts FLOPs versus regular convolution while keeping **higher FLOPS** than DWConv/GConv by reducing memory access.
- **FasterNet family** built on PConv, fast on many device classes and effective for classification, detection and segmentation.
- **FasterNet-T0** — 2.8×/3.3×/2.4× faster than MobileViT-XXS on GPU/CPU/ARM while being 2.9% more accurate on ImageNet-1k.
- **FasterNet-L** — 83.5% top-1, on par with Swin-B, with 36% higher GPU throughput and 37% less CPU compute time.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Chen_et_al._2023_2303.03667.md`](references/papers/Chen_et_al._2023_2303.03667.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DWConv/GConv-based efficient nets → FasterNet; siblings: MobileNetV4, RepViT, StarNet, EfficientViT; contrast: CycleMLP and MobileViT, which trade latency for FLOPs).
