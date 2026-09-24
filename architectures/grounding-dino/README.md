# Grounding DINO

## Overview

Grounding DINO merges the DINO detector with **grounded pre-training** so a detector can be driven by **free-form text**: the user names a category (or a referring expression) and the model outputs boxes with the matching scores. Three modules do the work — a neck-like **feature enhancer** that fuses image and text features, **language-guided query selection**, and a **cross-modality decoder**.

- **Year:** 2024
- **Authors:** Liu et al. (Tsinghua University / IDEA Research)
- **Source:** arXiv:2303.05499 — *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*
- **Category:** DL/Detection

## Key Characteristics

- **Open-set / zero-shot detection**: categories are not fixed by the training label space; they come from the text prompt at inference.
- **Tight modality fusion** at three stages (neck, query initialization, head) rather than a late fusion — the reason text actually steers the detection.
- Dual backbones: a vision backbone (Swin Transformer) and a text backbone (BERT), with deformable self-attention plus image-to-text and text-to-image cross-attention in the enhancer.
- **Sub-sentence level** text features: each phrase/word is grounded separately, which makes long prompts and referring expressions work.
- Supports zero-shot transfer to referring expression comprehension (REC) tasks and open-vocabulary detection.
- Cost: two encoders and a multi-modal decoder make it heavier than a closed-set detector of the same resolution.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Liu_et_al._2024_2303.05499.md`](references/papers/Liu_et_al._2024_2303.05499.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (DINO → Grounding DINO; siblings: GLIP, OWL-ViT, DetCLIP; downstream: Grounded SAM, open-vocabulary segmentation).
