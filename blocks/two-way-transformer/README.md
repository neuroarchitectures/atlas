# Two-Way Transformer

## Design Philosophy

Prompt-aware segmentation needs *both* directions of conditioning: prompts should reshape image features, and image content should refine what the prompts mean. A lightweight decoder alternates prompt→image and image→prompt cross-attention so both streams co-update before mask upsampling.

## Functionality

- Two layers each of: prompt-to-image cross-attn (prompts as K/V... image queries) and image-to-prompt cross-attn (prompt queries), with MLPs.
- Final prompt tokens + upsampled image embedding → dynamic mask MLP → mask logits; multi-mask outputs possible.

## Used By

| Model | Role |
|-------|------|
| SAM | 256-d mask decoder |
| SAM 2 | Same decoder + memory attention feeding it |
| SAM-HQ | Retained; HQ-Output token routes to an upsampling path |

## Features

- **Tiny compute** — runs per prompt in real time; the heavy encoder runs once per image.
- **Bidirectional conditioning** — the core novelty over one-way conditional decoders.

## Evolution

- **Predecessor**: DETR decoder (image-side only); DETR-style promptable heads.
- **Successor**: SAM 3's detection decoder replaces it for concept segmentation; masked-attention decoders (Mask2Former) for panoptic.
