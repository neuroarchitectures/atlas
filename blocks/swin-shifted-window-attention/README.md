# Swin Shifted-Window Attention Block

## Design Philosophy

Self-attention runs inside fixed **local windows** (linear cost in image size), and every other block **shifts** the windows so information crosses window boundaries. The philosophy: ViT's global attention is O(n²) in pixels — too expensive for detection/segmentation resolutions. Windowed attention is linear, and the shifted-window trick lets non-adjacent windows communicate without global attention. Hierarchical (4 stages, halving resolution, doubling channels) so it drops into FPN-style heads.

## Functionality

- **Window partition**: Split the feature map into non-overlapping `M×M` windows.
- **W-MSA** (window multi-head self-attention): Standard MHA *within* each window. Cost: O(M²·HW/M²) = O(M²·n_patches) — linear in image size.
- **SW-MSA** (shifted window): Shift the partition by `M/2` so windows straddle the old boundaries → information crosses windows.
- **Alternating**: Layer `l` uses W-MSA, layer `l+1` uses SW-MSA, repeat.
- **Patch merging** (between stages): 2×2 → linear, halving resolution, doubling channels.

## Used By

| Model | Role |
|-------|------|
| Swin-Tiny/Small/Base | 4 stages, depths 2/2/6/2 (Tiny) |
| Swin-V2 | Scaled variant |
| Mask R-CNN (Swin backbone) | Backbone for detection/segmentation |

## Features

- **Linear in image size**: Windowed attention scales to high resolution.
- **Cross-window communication**: The shifted-window trick, no global attention needed.
- **Hierarchical**: 4 stages with pyramid features → drops into FPN heads.

## Evolution

- **Predecessor**: ViT (global attention, O(n²)); local attention methods.
- **Successor**: Swin-V2; ConvNeXt (shows conv can match Swin without attention); hierarchical ViT variants (CSWin, etc.).
