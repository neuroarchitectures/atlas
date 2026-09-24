# Perception Encoder (PE)

## Overview

Perception Encoder (PE) is Meta FAIR's unified visual encoder family for images **and** video, built on one finding stated in its title: **the best visual embeddings are not at the output of the network**. A purely **contrastive** core (PE-Core) — a refined image recipe plus a scaled video data engine — produces embeddings that are strong *everywhere*, provided you read them from the **intermediate layers**: PE-Language aligns mid-layer features to an LLM for reading/QA; PE-Spatial aligns attention for dense tasks (detection, depth, tracking). One encoder family serves *looking* (zero-shot classification/retrieval), *reading* (document/video QA), and *spatial action* (dense prediction).

- **Year:** 2025
- **Authors:** Bolya et al. (Meta FAIR)
- **Source:** arXiv:2504.13181 — *Perception Encoder: The best visual embeddings are not at the output of the network*
- **Category:** DL/Vision-Foundation

## Key Characteristics

- **Contrastive-only core (PE-Core)**: refined image-text pretraining + large-scale video training with a data engine of synthetic and human-annotated captions — no multi-objective pretraining soup.
- **Intermediate-layer embeddings**: features for language alignment come from middle layers, not the final projection — the paper's central empirical finding.
- **PE-Language**: mid-layer features + adapter feed an LLM (8B) — DocVQA 94.6, InfographicVQA 80.9, PerceptionTest 82.7.
- **PE-Spatial**: attention-level alignment for dense tasks — COCO **66.0 box mAP** (SOTA at publication).
- **Strong zero-shot**: ImageNet robustness suite average 86.6; Kinetics-400 video zero-shot 76.9.
- **Family sizes**: L / H / G (ViT-L up to ViT-g, ~1B-class), with distilled variants for smaller budgets.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Bolya_et_al._2025_2504.13181.md`](references/papers/Bolya_et_al._2025_2504.13181.md); official code/models at [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models)

## Related Architectures

See `architecture.md` → Evolution section (CLIP/SigLIP and DINOv2/I-JEPA → PE; contemporary sibling: DINOv3).
