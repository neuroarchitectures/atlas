# Temporal Consistency Loss

## Design Philosophy

Per-frame depth prediction flickers: independent frames disagree at corresponding points. Penalize the *temporal gradient* of depth — the change in predicted depth at corresponding pixels across frames — directly suppressing flicker and drift, instead of relying on smooth per-frame predictions to somehow average out.

## Functionality

- `L_temp = ‖D_t(p) − D_{t+1}(p')‖` over correspondences p ↔ p' (estimated flow or fixed camera assumption); combined with per-frame photometric/depth losses.
- Video Depth Anything pairs it with a lightweight spatio-temporal attention head over a frame window + key-frame scheduling for long videos.

## Used By

| Model | Role |
|-------|------|
| Video Depth Anything | Temporal gradient suppression over frame windows |

## Features

- **Targets the failure mode directly** — flicker, not accuracy, is the enemy in video.
- **Pairs with any per-frame head** — an additive training term.

## Evolution

- **Predecessor**: sliding-window averaging, optical-flow-warping consistency losses.
- **Related**: temporal-bev-attention (architectural temporal memory) — the architectural alternative to a loss-level fix.
