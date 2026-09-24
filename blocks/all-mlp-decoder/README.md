# All-MLP Segmentation Decoder

## Design Philosophy

Attention-based segmentation decoders spend compute re-mixing features that the hierarchical encoder already mixed. Replace the decoder entirely with MLPs: upsample each encoder level to 1/4 resolution, concatenate, and apply one MLP head — positional information is supplied by the encoder's Mix-FFN convs, not the decoder.

## Functionality

- Per level `i`: `Conv1x1 → bilinear upsample to 1/4`.
- `Concat(L1..L4) → Conv3x3 (fuse) → Conv1x1 → Conv1x1 → class logits → upsample to full res`.

## Used By

| Model | Role |
|-------|------|
| SegFormer | Semantic segmentation mask from the 4-level hierarchical transformer encoder |

## Features

- **No decoder attention, no positional interpolation fragility** — works at test resolutions the encoder never saw.
- **Linear-complexity head** over already-hierarchical features.

## Evolution

- **Predecessor**: SETR (transformer decoder), FPN-style decoders.
- **Successor**: DPT-decoder (depth), Mask2Former's masked attention decoders for panoptic tasks.
