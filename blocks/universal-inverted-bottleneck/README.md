# Universal Inverted Bottleneck (UIB)

## Design Philosophy

Instead of hand-picking one inverted-bottleneck variant per model, make the block itself *searchable*: an inverted bottleneck with two optional depthwise convolutions whose presence/absence is a NAS decision. One block definition then degenerates into inverted residual, ConvNeXt block, FFN (conv-less), or ExtraDW depending on the flags.

## Functionality

- Optional PW-expand → optional DW-conv3 → optional DW-conv-k → optional PW-project, with residuals around the active middle.
- Two-phase NAS (block choice first, then per-block expansion tuning) selects the configuration per stage.

## Used By

| Model | Role |
|-------|------|
| MobileNetV4 | UIB blocks chosen per stage by NAS; hybrid variants add Mobile MQA attention |

## Features

- **One search space, four block types** — IB, ConvNeXt-style, pure FFN, ExtraDW all appear as flag settings.
- **Fusion of mobile block literature** into a single trainable superset.

## Evolution

- **Predecessor**: inverted-residual, mbconv, convnext-block.
- **Successor**: hybrid UIB + Mobile MQA blocks in MobileNetV4's attention variants.
