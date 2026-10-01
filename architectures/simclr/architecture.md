# Architecture: SimCLR

## Motivation

Self-supervised learning needs to define positive and negative pairs. SimCLR uses data augmentation to create positive pairs and in-batch negatives, with a contrastive loss.

## Core Idea

Apply two random augmentations to each image -> two views. Encode both with a shared encoder. Project with an MLP head. NT-Xent loss: pull positive pairs together, push negatives apart.

## Architecture

### Overview

![simclr architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | View 1 (Augment) | `conv` | random crop + color jitter |
| 2 | View 2 (Augment) | `conv` | different random aug |
| 3 | Encoder (ResNet) | `linear` | shared weights, 512-dim |
| 4 | Projection MLP | `linear` | 2-layer, 128-dim output |
| 5 | Contrastive Loss | `output` | NT-Xent (normalized temp-scaled cross entropy) |

</details>

### Components

1. **Data augmentation** — Random crop + color jitter + Gaussian blur. Two views per image. 2. **Encoder** — A ResNet (or ViT) that produces a 512-dim representation. Shared weights for both views. 3. **Projection head** — A 2-layer MLP that projects to a 128-dim space where contrastive loss is applied. Discarded after pretraining. 4. **NT-Xent loss** — Normalized temperature-scaled cross entropy. For each positive pair, all other in-batch samples are negatives. Temperature tau controls sharpness.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Apply two random augmentations to each image -> two views. Encode both with a shared encoder. Project with an MLP head. NT-Xent loss: pull positive pairs together, push negatives apart.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: MoCo, InstDisc. Successor: BYOL, MoCo v2/v3, DINO.

## References

- Chen et al. 2020
