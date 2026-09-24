# SAM 3

## Overview

SAM 3 moves from "segment the thing I point at" to **promptable concept segmentation (PCS)**: given a concept — a short noun phrase or an image exemplar — find and track **every instance** of it in an image or video. A shared perception encoder and a single unified detector-tracker handle both images and video, and a presence head decides whether the concept is present at all.

- **Year:** 2025
- **Authors:** Carion et al. (Meta Superintelligence Labs)
- **Source:** arXiv:2511.16719 — *SAM 3: Segment Anything with Concepts*
- **Category:** DL/Segmentation

## Key Characteristics

- **Two prompt modalities**: text (noun phrase) **and** image exemplars, which can be combined; a dedicated exemplar encoder handles the visual prompt.
- **Unified detector-tracker** — one architecture produces detections and tracks masks over video instead of running detection per frame and associating afterwards.
- **Presence head** — explicitly scores whether the concept appears, which is what makes open-vocabulary prompts usable without a fixed label set (and what "no instance" queries require).
- **Shared perception encoder** across image and video, so the same weights serve both modalities.
- **Joint image + video objective**, keeping SAM's original visual (point/box) segmentation ability while adding concept-level understanding.
- Evaluated on the **SA-Co** benchmark built for concept segmentation.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Carion_et_al._2025_2511.16719.md`](references/papers/Carion_et_al._2025_2511.16719.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (SAM → SAM 2 → SAM 3; siblings: Grounding DINO, OWL-ViT, EfficientSAM; downstream: concept tracking, open-vocabulary video segmentation).
