# Architecture: DINOv2

## Motivation

Self-supervised vision models trained on small datasets (ImageNet) are limited. DINOv2 scales up data (142M curated images), model size (1B params), and combines DINO self-distillation + iBOT masked prediction for universal features.

## Core Idea

A ViT is trained with two losses: DINO (self-distillation between global views) and iBOT (masked patch prediction). Student network learns from teacher (EMA-updated). Multi-crop augmentation. Trained on LVD-142M curated dataset.

## Architecture

### Overview

![dinov2 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (4 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3x224x224 |
| 2 | Patch Embedding | `identity` | 16x16 patches |
| 3 | ViT Encoder | `linear` | up to 40 layers, 1024 dim |
| 4 | Features | `output` | universal representations |

</details>

### Components

1. **Self-distillation (DINO)** — student matches teacher on global views. 2. **Masked prediction (iBOT)** — predict masked patches. 3. **Teacher EMA** — exponential moving average student. 4. **Multi-crop** — 2 global + 8 local crops. 5. **LVD-142M** — curated, deduplicated dataset. 6. **KoLeo regularization** — uniformity of feature distribution. 7. **Universal features** — work across tasks without fine-tuning.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A ViT is trained with two losses: DINO (self-distillation between global views) and iBOT (masked patch prediction). Stud...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DINO, iBOT, MoCo. Successor: DINOv3.

## References

- Oquab et al. 2023
