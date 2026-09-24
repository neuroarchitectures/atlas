# Intermediate-Layer Feature Tap

## Design Philosophy

Final-layer features are a compromise: language tasks want late semantics, dense tasks want mid-level spatial detail. Instead of one projection head, tap *multiple intermediate layers* — the tap depth per task head becomes a probed design choice, exposing the representation spectrum of one frozen encoder.

## Functionality

- Each alignment head reads from a chosen layer: PE-Language taps mid/late layers into an LLM adapter; PE-Spatial taps earlier layers + attention maps into dense prediction heads.
- Tap selection via linear probing during development; frozen thereafter.

## Used By

| Model | Role |
|-------|------|
| Perception Encoder (PE) | One ViT-L/H/G core serving both language-alignment and spatial-alignment heads |

## Features

- **One encoder, many readouts** — avoids per-task fine-tuned encoders.
- **Layer choice = task knob**, made explicit and probed.

## Evolution

- **Predecessor**: DINOv2/ViT-Adapter multi-layer readouts; FPN-style taps in conv nets.
- **Related**: vit-dense-adapter — adapter modules as an alternative to raw taps.
