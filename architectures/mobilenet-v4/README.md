# MobileNetV4

## Overview

Efficient mobile models are usually Pareto optimal on *one* accelerator and mediocre on the rest — a model tuned for a mobile CPU often loses badly on a DSP, an EdgeTPU or the Apple Neural Engine. MobileNetV4 targets that **universality** directly: search over a unified block that subsumes several well-known micro-architectures, add an attention block built for accelerators, and refine the NAS recipe so the resulting suite is mostly Pareto optimal across every device class tested.

- **Year:** 2024
- **Authors:** Qin et al. (Google)
- **Source:** arXiv:2404.10518 — *MobileNetV4 — Universal Models for the Mobile Ecosystem*
- **Category:** DL/Efficient CNN

## Key Characteristics

- **Universal Inverted Bottleneck (UIB)** — a single flexible search block that unifies Inverted Bottleneck, ConvNeXt, FFN and a new **Extra Depthwise (ExtraDW)** variant, by adding two optional depthwise convolutions; gives spatial/channel mixing flexibility, optional receptive-field extension, and better compute efficiency.
- **Mobile MQA** — an attention block designed for mobile accelerators, over **39% faster** than Multi-Head Attention there.
- **Refined two-phase NAS recipe** — coarse then fine-grained search, improving search effectiveness.
- **Mostly Pareto optimal across all device classes** — mobile CPUs, DSPs, GPUs, Apple Neural Engine, Google Pixel EdgeTPU; the paper states this uniformity is not found in any other tested model.
- **New distillation technique** — MNv4-Hybrid-Large reaches **87% ImageNet-1K** at 3.8 ms on a Pixel 8 EdgeTPU.
- **Performance modeling included** — analysis of *how* the cross-device performance was achieved, not just the numbers.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Qin_et_al._2024_2404.10518.md`](references/papers/Qin_et_al._2024_2404.10518.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MobileNetV1/V2/V3 → MobileNetV4; siblings: RepViT, StarNet, FasterNet, EfficientViT, GhostNet; contrast: MobileNetV2 already in catalog but not accelerator-universal).
