# ViT Dense Prediction Adapter

## Design Philosophy

Plain ViTs are pretrained for global representation, not dense prediction — and fine-tuning them for every dense task destroys the pretraining investment. ViT-Adapter adds a *parallel* adapter branch: a spatial prior module injects CNN-style local context into transformer layers, and a multi-scale extractor emits FPN-like features — trained from scratch while the ViT stays frozen (or lightly tuned).

## Functionality

- Spatial Prior Module: small conv stack → context features fused into transformer layers at interaction points.
- Interaction: single/multi-layer forward interaction injects SPM context into the ViT stream.
- Multi-Scale Feature Extractor: deconv-based extractor branches off ViT features into P2–P5 pyramid.

## Used By

| Model | Role |
|-------|------|
| ViT-Adapter | Frozen multimodal-pretrained plain ViT (e.g., MIM-pretrained) + randomly initialized adapter for detection/segmentation |

## Features

- **Pretraining-free adaptation** — dense capability without touching the backbone weights.
- **Bridges plain ViT and hierarchical heads** — the missing FPN plumbing.

## Evolution

- **Predecessor**: ViT-FPN (naive reshape heads), CPVT/PVT hierarchical redesigns.
- **Related**: intermediate-layer-tap (read out layers directly), dpt-decoder (reassembly alternative).
