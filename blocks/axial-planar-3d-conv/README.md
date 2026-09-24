# Axial-Planar 3D Convolution

## Design Philosophy

Full 3D convolutions over a cost volume (H × W × D) are expensive and slowly-growing in receptive field. Decouple the volume into two axes: separate *spatial* 3D convolutions (over image plane) and *disparity-axis* 3D convolutions, each seeing the whole volume in its own dimension at manageable cost.

## Functionality

- Attentive Hybrid Cost Volume filtering: spatial 3D convs aggregate context per disparity slice; disparity 3D convs aggregate along the depth axis; results combined.
- Used inside the cost-volume filtering stage before soft-argmax regression.

## Used By

| Model | Role |
|-------|------|
| FoundationStereo | Cost volume filtering of the 4D (H,W,D,features) volume |

## Features

- **Larger receptive field per FLOP** than isotropic 3D convs.
- **Axis-specialized filters** — spatial context and depth context learned separately.

## Evolution

- **Predecessor**: PWC-Net/RAFT-style 2D filtering + GRU iterations; GC-Net full 3D CNNs.
- **Related**: axially decomposed attention (Swin, CSWin) — the same axis-decomposition idea in attention form.
