# Cross-Shaped Window Attention

## Design Philosophy

Square windows (Swin) limit each token to a 2D neighborhood; axial attention is cheap but 1D. CSWin computes self-attention in parallel over *horizontal and vertical stripes* forming a cross-shaped window — split heads by orientation and vary stripe width per stage for multi-scale receptive fields.

## Functionality

- Half the heads attend H-Stripes, half V-Stripes; stripe width w ∈ {1, 2, 4, 7} scheduled across stages.
- Locally-Enhanced Positional Encoding (LePE): depthwise conv on V adds local position bias, keeping arbitrary test resolution.

## Used By

| Model | Role |
|-------|------|
| CSWin | 4-stage hierarchical transformer backbone |

## Features

- **Cross-shaped receptive field** — longer reach than square windows at the same token budget.
- **Per-stage stripe width** — multi-scale context growth across the hierarchy.

## Evolution

- **Predecessor**: swin-shifted-window-attention, Axial attention.
- **Related**: multi-axis-attention (blocked + dilated grid), area-attention (strip areas in YOLO).
