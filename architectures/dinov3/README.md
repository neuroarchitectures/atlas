# DINOv3

## Overview

DINOv3 is a **self-supervised vision foundation model** whose frozen features beat specialized SOTA on dense tasks (depth estimation, segmentation, correspondence) without any fine-tuning. The enabling idea is **Gram anchoring**: a loss term that anchors the Gram matrix of the student's patch features to that of a small teacher trained on a small curated dataset, which prevents the dense-feature degradation that previously forced web-scale pretraining to rely on expensive dataset curation.

- **Year:** 2025
- **Authors:** Siméoni et al. (Meta AI / INRIA)
- **Source:** arXiv:2508.10104 — *DINOv3*
- **Category:** DL/Vision-Foundation

## Key Characteristics

- **Gram anchoring**: second-order statistics (patch-feature Gram matrices) of the student are anchored to a small teacher trained on a small dataset — this decouples dense-feature quality from dataset scale and curation, enabling pretraining on **uncurated** web-scale image data (~1.2B images).
- **Frozen features**: a single model, frozen, exceeds specialized and fine-tuned baselines on dense prediction benchmarks across domains — natural images to aerial/satellite imagery.
- **Model family**: ViT-S / B / L / H / **7B** (≈21M–6.8B params, patch 16); smaller models obtained by **distillation** from the ViT-7B teacher.
- **High resolution / long context**: trained at high resolution and robust at inference resolutions well beyond 1000px — long token sequences degrade gracefully instead of collapsing.
- **Post-hoc flexibility**: optional CLIP-style text alignment and resolution adaptation applied *after* pretraining, so one pure-vision core serves dense, retrieval, and zero-shot tasks.
- **License**: code Apache-2.0; **weights under the custom DINOv3 License** (commercial use permitted for organizations under ~$1B revenue) — unlike DINOv2's Apache-2.0 weights.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Siméoni_et_al._2025_2508.10104.md`](references/papers/Siméoni_et_al._2025_2508.10104.md)

## Related Architectures

See `architecture.md` → Evolution section (DINO 2021 → DINOv2 → DINOv3; siblings: I-JEPA, CLIP, Perception Encoder; downstream users: VGGT-family geometry models). Note: **unrelated to `dino/`** in this repo, which is the same-named DETR-family *detector*.
