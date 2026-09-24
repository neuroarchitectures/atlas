# DeiT

## Overview

The original ViT was trained on a 300M-image private dataset and concluded that transformers "do not generalize well when trained on insufficient amounts of data". DeiT removes that precondition: a vision transformer trained on **ImageNet alone**, on a single 8-GPU node in two to three days, competitive with convnets of similar size and efficiency — plus a distillation strategy specific to transformers.

- **Year:** 2020
- **Authors:** Touvron et al. (Facebook AI Research, Sorbonne University)
- **Source:** arXiv:2012.12877 — *Training data-efficient image transformers & distillation through attention*
- **Category:** DL/ViT

## Key Characteristics

- **No external data** — ImageNet is the sole training set; the models contain **no convolutional layer** and still match the state of the art.
- **Single-node budget** — 53 hours of pre-training (optionally 20 hours of fine-tuning); 4 GPUs in three days for the smaller variants.
- **Token-based distillation** — a strategy specific to transformers, denoted by DeiT, which the paper shows advantageously replaces the usual distillation; introduces a **distillation token** alongside the class token.
- **Training recipe as contribution** — the ablation details the hyper-parameters and key ingredients for successful training, notably **repeated augmentation**, since the architecture itself is nearly unchanged from Dosovitskiy et al.
- **DeiT-S and DeiT-Ti** — fewer parameters, positioned as the counterparts of ResNet-50 and ResNet-18.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Touvron_et_al._2020_2012.12877.md`](references/papers/Touvron_et_al._2020_2012.12877.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (ViT (JFT-300M) → DeiT; siblings: Swin, ViT variants, MAE; successors: DeiT III and the many ViT training-recipe follow-ups; contrast: convnets such as ResNet-50 at equal parameter count).
