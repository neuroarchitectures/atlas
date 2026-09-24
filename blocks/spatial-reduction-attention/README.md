# Spatial-Reduction Attention

## Design Philosophy

Self-attention cost explodes on high-resolution feature maps, where spatial detail lives but most keys are redundant. Reduce K and V *spatially* (projection to fewer tokens) before attention; Q stays full-resolution. Cost drops linearly in the reduction ratio.

## Functionality

- PVT (SRA): `K, V ← R(K), R(V)` via conv/stride projection to r× fewer tokens.
- MViTv2 variant (pooling attention): pool keys/values with strided local aggregation + a residual pooling connection; the pooling stride schedule shrinks across stages.

## Used By

| Model | Role |
|-------|------|
| PVT | 4-stage shrinking pyramid with per-stage reduction ratios |
| MViTv2 | Pooling attention per stage of the multi-scale video/image hierarchy |

## Features

- **Keep Q dense** — every output position still gets an attended summary.
- **Stride schedule = compute budget** across stages.

## Evolution

- **Predecessor**: Pyramidal convs; Swin's windowing as the alternative cost cut.
- **Successor**: multi-axis-attention and area-attention achieve long range without discarding keys entirely.
