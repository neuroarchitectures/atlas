# Cross-Stage-Partial (CSP) Stage

## Design Philosophy

Duplicate gradient information flows through dense transition stages. CSP splits the channels: only part goes through the conv stack, the rest bypasses via concat — cutting redundant computation and gradients while keeping accuracy.

## Functionality

- Split input channels into two parts; part 1 → dense/bottleneck conv stack; concat with part 2 → transition conv.
- CSPNeXt variant adds large-kernel depthwise convs in the stack; C3K2 (YOLO11) uses two small bottleneck convs instead of one large.

## Used By

| Model | Role |
|-------|------|
| RTMPose (CSPNeXt) | Backbone stages with large-kernel depthwise convs |
| YOLOv11 (C3K2) | CSP bottleneck variant in backbone/neck |
| YOLOv12 (R-ELAN) | ELAN-style aggregation descendant with block residuals |

## Features

- **Gradient-flow diversification** — bypass path carries raw features to the concat.
- **FLOP reduction** without depth loss — the standard trick of the YOLO/RTMPose family.

## Evolution

- **Predecessor**: CSPNet (2020); DenseNet dense-connectivity (which CSP prunes).
- **Successor**: ELAN (YOLOv7), R-ELAN (YOLOv12) — aggregate more branches per stage.
