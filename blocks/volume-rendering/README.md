# Differentiable Volume Rendering

## Design Philosophy

Turn a continuous density+color field into an image by marching rays and *classically compositing* — alpha-compositing accumulation along each ray, the way volume graphics always worked — but done in differentiable autodiff so a photometric reconstruction loss trains the field itself.

## Functionality

- Per ray: sample M points; accumulate `C = Σ T_i α_i c_i`, `T_i = Π_{j<i} (1-α_j)`, where α_i = 1 − exp(−σ_i δ_i).
- Sample stratification (coarse + fine importance sampling) drives quality.

## Used By

| Model | Role |
|-------|------|
| NeRF | Photometric reconstruction from posed images only; MLP outputs density + view-dependent RGB |

## Features

- **End-to-end differentiable classic graphics** — no 3D supervision needed.
- **View-dependent color** — SH/specular effects via direction-conditioned outputs.

## Evolution

- **Predecessor**: classic volume rendering (Levoy 1988), DeepVoxels.
- **Successor**: instant-NGP (hash encoding), 3d-gaussian-primitives + tile-based-splatting (explicit primitives replace the field).
