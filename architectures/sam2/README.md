# SAM 2

## Overview

Promptable visual segmentation extended from images to **video**. A Hiera image encoder produces per-frame features; a memory mechanism (memory encoder, memory bank, memory attention) conditions the current frame on past frames and past prompts, so a single click can produce a consistent **masklet** through an entire clip.

- **Year:** 2024
- **Authors:** Ravi et al. (Meta AI, FAIR)
- **Source:** arXiv:2408.00714 — *SAM 2: Segment Anything in Images and Videos*
- **Category:** DL/Segmentation

## Key Characteristics

- **Unified image + video model**: the image case is the single-frame special case; streaming inference processes frames left-to-right.
- **Memory attention**: transformer blocks with self-attention, cross-attention to the memory bank (recent + prompted frames), and RoPE on spatial positions.
- **Memory bank** stores object memories for the last N frames and all prompted frames, plus lightweight **object pointer** tokens used for re-identification after occlusion.
- **Occlusion head** predicts whether the target object is visible in the current frame.
- Prompt encoder accepts points, boxes, and masks; multiple clicks refine the same object.
- Data engine produced SA-V (≈50.9K videos, ≈642.6K masklets); model/data released under permissive licenses.
- Limitation for on-device use: accuracy depends on memory-bank size and encoder scale; long-term occlusion, fast motion, and visually similar instances remain failure modes.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Ravi_et_al._2024_2408.00714.md`](references/papers/Ravi_et_al._2024_2408.00714.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (SAM → SAM 2 → SAM 2.1 / SAM 3; relation to Hiera, Perceiver-style prompt decoders, and XMem/STCN-style memory networks).
