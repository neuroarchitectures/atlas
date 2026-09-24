# Sparse/Dense Prompt Encoder

## Design Philosophy

Make one segmentation model steerable at *run time* by encoding interaction prompts — points, boxes, masks, text — into the model's token space. Sparse prompts become positional + type embeddings; dense prompts (masks) are conv-embedded and summed into the image embedding.

## Functionality

- Sparse: each point/box corner → learned positional embedding + type embedding; text via a text encoder's features.
- Dense: mask → small CNN → embedding summed with image embedding.
- Output: 256-d token sequence consumed by the mask decoder.

## Used By

| Model | Role |
|-------|------|
| SAM | Points/boxes/masks prompt encoding |
| SAM 2 | Same encoder over frames (points/boxes/masks as memory prompts) |
| SAM 3 | Visual exemplar + text (noun phrase) prompts |
| SAM-HQ | Retained unchanged; HQ quality comes from feature fusion instead |

## Features

- **Interactive segmentation** — the mechanism that makes prompting a first-class interface.
- **Modality-flexible** — any prompt type maps to one token space.

## Evolution

- **Predecessor**: conditional segmentation heads (pointer networks, VLT).
- **Successor**: SAM 3 adds concept-level (text/exemplar) prompts beyond spatial ones.
