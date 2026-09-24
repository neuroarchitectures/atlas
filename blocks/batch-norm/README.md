# BatchNorm (Batch Normalization)

## Design Philosophy

Internal covariate shift — the change in layer-input distributions as earlier layers train — makes deep networks hard to optimize. BatchNorm's answer is to **re-normalize every layer's activations using the current mini-batch statistics**, which smooths the loss landscape, permits much higher learning rates, and acts as a regularizer. It is the reason very deep CNNs became trainable.

## Functionality

During training: `BN(x) = γ ⊙ (x − μ_B) / √(σ²_B + ε) + β`, with `μ_B, σ²_B` computed **per channel over the batch and spatial positions**.

- **Running statistics** of mean and variance are accumulated with momentum and used at inference.
- Learnable per-channel affine `γ, β` (restore the representational capacity removed by normalization).
- Applied typically as Conv → BN → activation; bias in the conv becomes redundant.

## Used By

| Model | Role |
|-------|------|
| ResNet / ResNeXt | Conv-BN-ReLU in every residual branch |
| CNN backbones (RT-DETR, Deformable DETR, RTMPose's CSPNeXt) | Normalization inside the backbone |
| MobileNet / EfficientNet | BN after every depthwise and pointwise conv |
| YOLO family | Conv-BN-SiLU (or BN after conv) blocks |

## Features

- **Higher learning rates and faster convergence**; reduces sensitivity to initialization.
- **Regularizing effect** — batch statistics add noise, often replacing some dropout.
- **Batch-size dependent**: small batches (or per-GPU micro-batches) degrade quality; hence SyncBatchNorm for multi-GPU training and **BatchNorm freezing** at small batch.
- **Inference/train mismatch**: relies on running statistics, which is why it is unsuitable for RNNs and awkward for transformers.

## Evolution

- **Predecessor**: input whitening / no normalization (pre-2015 deep CNNs needed careful init).
- **Itself**: BatchNorm (Ioffe & Szegedy, 2015).
- **Successors/variants**: LayerNorm (batch-independent, for sequences), GroupNorm (batch-free, for detection/segmentation with small batches), SyncBatchNorm, BatchNorm freezing, RMSNorm.
