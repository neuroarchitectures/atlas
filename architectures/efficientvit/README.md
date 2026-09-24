# EfficientViT

## Overview

High-resolution dense prediction (segmentation, super-resolution, Segment-Anything features) needs **high resolution and strong context**, which is exactly the combination that expensive models provide and affordable ones do not. Prior efficient models reach that combination with heavy softmax attention, hardware-inefficient large-kernel convolutions, or complicated topologies. EfficientViT gets global receptive field and multi-scale learning from operations that are cheap on real hardware instead.

- **Year:** 2022
- **Authors:** Cai et al. (MIT HAN Lab, Songshuge)
- **Source:** arXiv:2205.14756 — *EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction*
- **Category:** DL/Efficient ViT

## Key Characteristics

- **Multi-scale linear attention** — the core module; achieves global receptive field *and* multi-scale learning with only lightweight, hardware-efficient operations.
- **Avoids the three usual costs** — no heavy softmax attention, no hardware-inefficient large-kernel convolution, no complicated topology.
- **Explicitly for dense prediction, not classification** — the paper notes that porting classification-efficient architectures is unsuitable, because dense prediction needs both resolution and context.
- **Large measured speedups at equal or better quality** — up to **13.9×** and **6.2×** GPU latency reduction over SegFormer and SegNeXt on Cityscapes without performance loss; up to **6.4×** speedup over Restormer for super-resolution with +0.11 dB PSNR; **48.9×** higher throughput than SAM on A100 with slightly better zero-shot instance segmentation on COCO.
- **Verified across hardware tiers** — mobile CPU, edge GPU and cloud GPU.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Cai_et_al._2022_2205.14756.md`](references/papers/Cai_et_al._2022_2205.14756.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (SegFormer, SegNeXt, Restormer → EfficientViT; siblings: MobileNetV4, RepViT, FasterNet; contrast: softmax-attention high-resolution models).
