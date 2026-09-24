# Multi-Axis Attention

## Design Philosophy

Global attention needs linear cost at arbitrary resolution. MaxViT composes two sparse attention patterns: *blocked* local attention (windowed, dense within) plus *dilated/strided* global attention (sparse tokens spanning the whole map) — together giving every token both a dense neighborhood and a global reach at O(n) cost.

## Functionality

- Blocked local attn: non-overlapping windows (like Swin).
- Grid/dilated global attn: partition the map into a coarse grid; each cell's tokens attend globally along their grid line.
- Alternated with MBConv in every MaxViT block.

## Used By

| Model | Role |
|-------|------|
| MaxViT | MBConv → blocked local attn → grid global attn per block, hierarchical stages |

## Features

- **Global-local at linear cost** and *any* input resolution (window sizes fixed).
- **Compose, don't choose** — local and global patterns are complements, not alternatives.

## Evolution

- **Predecessor**: Swin (local only), axial attention (one axis at a time).
- **Related**: cswin-attention (cross-shaped stripes), area-attention (strip windows).
