# Ghost Module

## Design Philosophy

Most feature maps are redundant: many channels are similar to each other. Instead of paying full convolution cost for all of them, generate a small set of *intrinsic* features with a regular convolution, then manufacture the remaining "ghost" features with cheap linear transforms (depthwise convolutions) — reproducing full feature maps at a fraction of the FLOPs and memory access.

## Functionality

- Primary conv produces `m` intrinsic features `Y' = X * W`.
- Each cheap linear op `Φ_i` (3×3/5×5 depthwise conv per ghost feature) yields `s-1` ghost features; output = `Concat(Y', Φ(Y'))` with `s = n/m` intrinsic-to-ghost ratio.

## Used By

| Model | Role |
|-------|------|
| GhostNet | Ghost bottlenecks: intrinsic + ghost concat, optional SE; stem conv is a Ghost module |

## Features

- **Same output shape, fewer ops** — downstream blocks see full-width feature maps.
- **Memory-access aware** — cheap ops reduce both FLOPs and access cost on mobile hardware.

## Evolution

- **Predecessor**: depthwise-separable conv — removes redundancy *within* a spatial filter, not *between* channels.
- **Successor**: GhostNet V2 (DFLA); partially superseded by partial-convolution (FasterNet), which removes computation instead of synthesizing it.
