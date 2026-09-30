# Architecture: RetinaNet

## Motivation

Single-stage detectors (YOLO, SSD) are faster than two-stage (Faster R-CNN) but less accurate. The key issue is extreme class imbalance: most anchors are background, causing the detector to be overwhelmed by easy negatives. Focal Loss solves this by down-weighting easy examples.

## Core Idea

A Feature Pyramid Network (FPN) backbone extracts multi-scale features. Two subnetworks: (1) classification subnet predicts class probabilities with Focal Loss, (2) regression subnet predicts box offsets. Focal Loss: FL(p) = -α(1-p)^γ log(p), where the (1-p)^γ term down-weights easy (high-confidence) examples, focusing training on hard examples.

## Architecture

### Overview

![retinanet architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [1, 3, 800, 800] |
| 2 | Backbone (ResNet) | `custom` | type: resnet50 |
| 3 | FPN | `custom` | type: feature_pyramid, outChannels: 256, levels: [3, 4, 5, 6, 7] |
| 4 | Cls Subnet | `custom` | type: cls_subnet, outChannels: 256, numAnchors: 9, numClasses: 80 |
| 5 | Focal Loss | `custom` | type: focal_loss, alpha: 0.25, gamma: 2.0 |
| 6 | Reg Subnet | `custom` | type: reg_subnet, outChannels: 256, numAnchors: 9 |
| 7 | Detections | `output` |  |

</details>

A Feature Pyramid Network (FPN) backbone extracts multi-scale features. Two subnetworks: (1) classification subnet predicts class probabilities with Focal Loss, (2) regression subnet predicts box offsets. Focal Loss: FL(p) = -α(1-p)^γ log(p), where the (1-p)^γ term down-weights easy (high-confidence) examples, focusing training on hard examples.

### Components

2. **Backbone (ResNet)** (`custom`, scope: `backbone`) — Params: type: resnet50
3. **FPN** (`custom`, scope: `neck`) — Params: type: feature_pyramid, outChannels: 256, levels: [3, 4, 5, 6, 7]
4. **Cls Subnet** (`custom`, scope: `head`) — Params: type: cls_subnet, outChannels: 256, numAnchors: 9, numClasses: 80
5. **Focal Loss** (`custom`, scope: `head`) — Params: type: focal_loss, alpha: 0.25, gamma: 2.0
6. **Reg Subnet** (`custom`, scope: `head`) — Params: type: reg_subnet, outChannels: 256, numAnchors: 9

### Data Flow

The architecture processes input through a sequence of 7 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to RetinaNet.

## Evolution

RetinaNet introduced Focal Loss, bridging the accuracy gap between single-stage and two-stage detectors. Successors: FCOS (anchor-free), CenterNet, YOLOX, and modern DETR-based detectors. Focal Loss is now used in many detection and classification tasks with class imbalance.

## Source

- **Paper:** arXiv:1708.02002
- **Year:** 2017
- **Authors:** Lin et al.
