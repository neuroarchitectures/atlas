# SAM (Segment Anything)

## Overview

SAM reframes segmentation as a **promptable task**: given any prompt (points, boxes, masks, or text) the model returns a valid mask for *something*. It is trained on **SA-1B** — over 1 billion masks on 11M licensed, privacy-preserving images — via a data engine that loops model and annotation, which is what makes zero-shot transfer to new distributions work.

- **Year:** 2023
- **Authors:** Kirillov et al. (Meta AI, FAIR)
- **Source:** arXiv:2304.02643 — *Segment Anything*
- **Category:** DL/Segmentation

## Key Characteristics

- **Three-part architecture**: a heavy image encoder (ViT-H with MAE pretraining) computed **once per image**, and a lightweight **prompt encoder** + **mask decoder** that can then answer prompts cheaply and interactively.
- **Promptable** with sparse prompts (points, boxes) and dense prompts (masks), plus text; the same weights serve all prompt types.
- **Ambiguity-aware**: a single point can mean several things (a shirt vs. the person wearing it), so the decoder emits **multiple valid masks** with scores rather than forcing one answer.
- **Zero-shot transfer** to unfamiliar distributions and downstream tasks, often competitive with fully supervised results.
- SA-1B dataset release (1B masks / 11M images) is as much of the contribution as the model.
- Cost: the encoder is the dominant compute; real-time use needs encoder caching or a distilled variant.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Kirillov_et_al._2023_2304.02643.md`](references/papers/Kirillov_et_al._2023_2304.02643.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (interactive segmentation → SAM → SAM 2, EfficientSAM, FastSAM, MobileSAM; downstream: Grounded SAM, Maskformer-style pipelines).
