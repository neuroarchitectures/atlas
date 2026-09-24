# MobileSAM

## Overview

MobileSAM isolates SAM's actual bottleneck — the **ViT-H image encoder** — and replaces it with a tiny ViT trained by **decoupled distillation**. The prompt encoder and mask decoder are copied unchanged from SAM. The result is a drop-in replacement that keeps SAM's promptable behaviour at mobile latency.

- **Year:** 2023
- **Authors:** Zhang et al. (Samsung Research / Korea University)
- **Source:** arXiv:2306.14289 — *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications*
- **Category:** DL/Segmentation

## Key Characteristics

- **Decoupled distillation**: instead of jointly backpropagating through the whole teacher, distill the image encoder **on its own** (MSE between teacher and student embeddings), then fine-tune the (frozen-decoder) pipeline — this is the trick that makes training cheap.
- Trained on a tiny fraction of SA-1B (reported as **0.1%, ~11k images**) with **under 1%** of the compute of full SAM training.
- **Tiny ViT** image encoder; prompt encoder + mask decoder are reused verbatim, so prompts behave exactly like SAM.
- Drop-in: any SAM-based application can swap the encoder checkpoint.
- Trade-off: slightly lower mask quality than ViT-H SAM, largest gap on small/thin objects and hard boundaries.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhang_et_al._2023_2306.14289.md`](references/papers/Zhang_et_al._2023_2306.14289.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (SAM → MobileSAM / MobileSAMv2; siblings: FastSAM, EfficientSAM, EdgeSAM, SAM-HQ, SAM 2).
