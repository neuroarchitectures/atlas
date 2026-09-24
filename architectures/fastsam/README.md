# FastSAM

## Overview

FastSAM is a CNN-based alternative to SAM: it does **all-instance segmentation first** with a YOLOv8-seg model, then resolves the prompt by **selecting** the right candidate mask. The point is that instance segmentation and prompt answering can be separated — the model needs no prompt at inference for its heavy pass, so it is far cheaper than ViT-H SAM.

- **Year:** 2023
- **Authors:** Zhao et al. (Image and Video Analysis Lab, CASIA)
- **Source:** arXiv:2306.12156 — *Fast Segment Anything*
- **Category:** DL/Segmentation

## Key Characteristics

- **Two-stage, prompt-free heavy pass**: YOLOv8-seg produces mask candidates for the whole image in one forward pass; prompts are handled afterwards by cheap selection.
- **Prototype + mask-coefficient heads** (YOLACT-style) — masks are formed by linearly combining prototypes, which is much cheaper than a transformer mask decoder.
- **Prompt-guided selection** supports points, boxes, and text: the candidate whose mask/region best matches the prompt is returned.
- Trained on only a small fraction of SA-1B (reported as **2%, i.e. 1/50**) and still comparable to SAM on several settings; runs at a small fraction of SAM's cost.
- Trade-off: mask quality and boundary fidelity are below SAM, and thin/small instances are weaker; optional refinement (e.g. matting) recovers some of it.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhao_et_al._2023_2306.12156.md`](references/papers/Zhao_et_al._2023_2306.12156.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (YOLACT / YOLOv8-seg → FastSAM; siblings: MobileSAM, EfficientSAM, EdgeSAM, SAM-HQ; baseline: SAM).
