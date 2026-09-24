# Canonical Camera Space Transform

## Design Philosophy

Instead of making the network camera-aware, make it camera-*blind*: warp every input (or its labels) into one canonical camera via the known calibration, train on that stable domain, and de-canonicalize predictions at inference. The camera model never enters the network — metric depth and normal estimation become calibration-free.

## Functionality

- Train: rescale/warp images + labels to canonical intrinsics (focal length, principal point); for labels, transform the 3D quantities instead of the image when cheaper.
- Inference: predict in canonical space → de-canonicalize using the test camera's parameters.

## Used By

| Model | Role |
|-------|------|
| Metric3D | 8M-image metric depth + normals across heterogeneous cameras |

## Features

- **One model, any camera** — the alternative philosophy to camera-aware-depth-conditioning.
- **Label-space warping** avoids interpolation artifacts on the image side.

## Evolution

- **Predecessor**: per-camera fine-tuning; MiDaS-style scale-shift alignment.
- **Related**: scale-shift-invariant-loss — affine alignment as a loss instead of a warp.
