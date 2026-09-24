# EdgeSAM

## Overview

EdgeSAM targets on-device SAM. It uses **encoder-only distillation** (the student learns SAM's image embedding, keeping the heavy encoder out of the loop) but adds **prompt-in-the-loop distillation**: during training, prompts are sampled *dynamically from the region where teacher and student disagree*, and fed to the student through a **dynamic linear classifier** that converts prompt positions into tokens. The prompt-related machinery is deactivated at inference, so the shipped model is a plain encoder + lightweight decoder.

- **Year:** 2023
- **Authors:** Zhou et al.
- **Source:** arXiv:2312.06660 — *EdgeSAM: Prompt-In-the-Loop Distillation for SAM*
- **Category:** DL/Segmentation

## Key Characteristics

- **Prompt-in-the-loop** — prompts are generated from the teacher-student disagreement area, so distillation focuses compute where the student is actually wrong.
- **Dynamic linear classifier** — replaces the prompt encoder: generates prompt tokens on the fly and is switched off at inference.
- **Encoder-only distillation** — the expensive teacher encoder is not re-run per prompt during student training.
- **Lightweight mask decoder** — the decoder is distilled too, not just the encoder, which is what makes it usable on a phone.
- **On-device latency** — reported in the tens of milliseconds range on mobile hardware, versus seconds for SAM's ViT-H.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhou_et_al._2023_2312.06660.md`](references/papers/Zhou_et_al._2023_2312.06660.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (SAM → MobileSAM → EdgeSAM; siblings: EfficientSAM (SAMI pretraining), FastSAM (CNN), SAM-HQ (quality); successors: MobileSAMv2, RepViT-SAM).
