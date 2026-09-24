# SlowFast Two-Pathway Factorization

## Design Philosophy

Spatial semantics and temporal motion need different sampling rates. Factorize the video network into two pathways: a **Slow** pathway at low frame rate with wide channels (spatial appearance), and a **Fast** pathway at high frame rate with narrow channels (fine motion) — fused laterally at intermediate stages, not only at the end.

## Functionality

- Slow: ~8 frames, full channel width; Fast: ~32 frames, channels ≈ 1/8 of Slow.
- Lateral connections from Fast → Slow at multiple stages (conv to match temporal stride, or sampling).

## Used By

| Model | Role |
|-------|------|
| SlowFast Networks | Action recognition backbone; instantiated over ResNet/ResNeXt stages |

## Features

- **Asymmetric budget** — most compute on spatial semantics, small budget still captures rapid motion.
- **Multi-stage fusion** — motion evidence reaches every semantic level, not just the classifier.

## Evolution

- **Predecessor**: two-stream networks (separate RGB/optical-flow CNNs, late fusion only).
- **Successor**: video transformers (TimeSformer, ViViT) replace the two pathways with attention factorization; TubeViT-style mixed sampling.
