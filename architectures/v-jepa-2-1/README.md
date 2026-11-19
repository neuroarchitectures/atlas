# V-JEPA 2.1

## Overview

V-JEPA 2.1 solves a critical deficiency of V-JEPA 2: the lack of **dense, spatially-structured representations**. While V-JEPA 2 produces strong global video features (excellent for classification and QA), its patch-level tokens lack the spatial organization needed for dense downstream tasks like depth estimation, point tracking, and fine-grained robot manipulation. V-JEPA 2.1 introduces four key components — dense prediction loss, deep self-supervision, a multimodal tokenizer, and effective scaling — that unlock dense features while maintaining or improving global performance.

- **Year:** 2026
- **Authors:** Corentin Mur-Labadia et al. (Meta FAIR)
- **Source:** arXiv:2603.14482 — *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning*
- **Category:** DL/Self-Supervised Representation

## Key Characteristics

- **Dense prediction loss**: both visible and masked tokens participate in the training signal, encouraging explicit spatial and temporal grounding — unlike V-JEPA 2 which only predicts masked tokens.
- **Deep self-supervision**: self-supervised objectives are applied at multiple intermediate layers of the encoder, not just the output, producing hierarchical features.
- **Multimodal tokenizer**: unifies image and video training under a single tokenization scheme, enabling joint pretraining on both modalities.
- **Effective scaling**: scales both model capacity and training data simultaneously, achieving SOTA on dense prediction benchmarks.
- **Patch-token PCA shows clear object-background separation** — V-JEPA 2's PCA does not, demonstrating the dense loss is what creates spatial structure.
- **+20pp robot grasping success** when used with V-JEPA 2-AC, proving dense features are critical for manipulation.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Mur-Labadia_et_al._2026_2603.14482.md`](references/papers/Mur-Labadia_et_al._2026_2603.14482.md)

## Related Architectures

See `architecture.md` → Evolution section (V-JEPA → V-JEPA 2 → V-JEPA 2.1; sibling: LeVJEPA; contrast: V-JEPA 2 lacks dense features).
