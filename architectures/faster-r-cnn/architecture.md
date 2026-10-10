# Architecture: Faster R-CNN

## Motivation

Prior detectors (R-CNN, Fast R-CNN) rely on external region proposal methods (Selective Search), which are slow and cannot be trained jointly with the detector. Faster R-CNN introduces a learned Region Proposal Network (RPN) that shares convolutional features with the detection head, enabling end-to-end training and near-real-time inference.

## Core Idea

A backbone CNN extracts features. A Region Proposal Network (RPN) slides over the feature map, predicting objectness scores and bounding box offsets for anchors at each location. Top proposals are refined by a RoI pooling + classification head. The RPN and detection head share the backbone features, making the system fast and trainable end-to-end.

## Architecture

### Overview

![faster-r-cnn architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (11 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [1, 3, 800, 800] |
| 2 | Backbone CNN (ResNet) | `custom` | type: resnet50, outChannels: 256 |
| 3 | RPN 3x3 Conv | `custom` | type: rpn_conv, outChannels: 256, kernelSize: 3 |
| 4 | RPN Cls (objectness) | `custom` | type: rpn_cls, anchors: 9 |
| 5 | RPN Reg (bbox) | `custom` | type: rpn_reg, anchors: 9 |
| 6 | Proposal Layer | `custom` | type: proposal, nmsThreshold: 0.7, preNmsTopN: 12000 |
| 7 | RoI Pooling | `custom` | type: roi_align, outputSize: 7 |
| 8 | FC | `linear` | outFeatures: 1024, inFeatures: 25088 |
| 9 | Cls (classes) | `linear` | outFeatures: 81, inFeatures: 1024 |
| 10 | Reg (bbox) | `linear` | outFeatures: 320, inFeatures: 1024 |
| 11 | Detections | `output` |  |

</details>

A backbone CNN extracts features. A Region Proposal Network (RPN) slides over the feature map, predicting objectness scores and bounding box offsets for anchors at each location. Top proposals are refined by a RoI pooling + classification head. The RPN and detection head share the backbone features, making the system fast and trainable end-to-end.

### Components

2. **Backbone CNN (ResNet)** (`custom`, scope: `backbone`) — Params: type: resnet50, outChannels: 256
3. **RPN 3x3 Conv** (`custom`, scope: `rpn`) — Params: type: rpn_conv, outChannels: 256, kernelSize: 3
4. **RPN Cls (objectness)** (`custom`, scope: `rpn`) — Params: type: rpn_cls, anchors: 9
5. **RPN Reg (bbox)** (`custom`, scope: `rpn`) — Params: type: rpn_reg, anchors: 9
6. **Proposal Layer** (`custom`, scope: `rpn`) — Params: type: proposal, nmsThreshold: 0.7, preNmsTopN: 12000
7. **RoI Pooling** (`custom`, scope: `head`) — Params: type: roi_align, outputSize: 7
8. **FC** (`linear`, scope: `head`) — Params: outFeatures: 1024, inFeatures: 25088
9. **Cls (classes)** (`linear`, scope: `head`) — Params: outFeatures: 81, inFeatures: 1024
10. **Reg (bbox)** (`linear`, scope: `head`) — Params: outFeatures: 320, inFeatures: 1024

### Data Flow

The architecture processes input through a sequence of 11 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Faster R-CNN.

## Evolution

Faster R-CNN introduced the learned RPN, replacing external proposal methods. Successors: Mask R-CNN (instance segmentation), Cascade R-CNN (multi-stage), RetinaNet (single-stage alternative), and DETR (transformer-based). The RPN + RoI pattern became the standard for two-stage detection.

## Source

- **Paper:** arXiv:1506.01497
- **Year:** 2015
- **Authors:** Ren et al.
