# Convex Upsampling

## Design Philosophy

Dense regression runs at 1/8 resolution for cost; naive bilinear upsampling blurs boundaries. Convex upsampling recovers full resolution as a *softmax-weighted convex combination* of the 3×3 neighborhood of each coarse vector — learned, motion-adaptive interpolation with the weights predicted from the refinement network's hidden state.

## Functionality

- Per coarse pixel: predict 9 softmax weights from hidden state; output = Σ w_i · neighbors_i (applied per 1/8 → full-res factor, e.g., ×8 or per upscale step).

## Used By

| Model | Role |
|-------|------|
| RAFT | 1/8 → full resolution flow |
| RAFT-Stereo | Disparity upsampled only at the end |
| SEA-RAFT | Convex flow upsampling, 8× |
| GMFlow | Convex upsampling after propagation |

## Features

- **Sharp, structure-aware upscaling** — follows motion/depth boundaries.
- **Nearly free** — one conv for weights + weighted sum.

## Evolution

- **Predecessor**: bilinear/deconv upsampling in flow nets.
- **Related**: convex combination upscalers in matting/segmentation (PointRend-style adaptive sampling is the discrete cousin).
