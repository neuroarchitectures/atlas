# RPN (Region Proposal Network)

## Design Philosophy

A lightweight fully-convolutional network that slides over the backbone feature map and proposes candidate object regions (RoIs), sharing computation with the detection backbone. The philosophy: instead of external region proposal methods (Selective Search, ~2s/image), the RPN generates proposals in ~10ms by being part of the network, trained end-to-end with the detector.

## Functionality

- **Input**: Backbone feature map (e.g., FPN level).
- **3×3 conv**: A shared conv over the feature map.
- **Two 1×1 convs (heads)**:
  - **Objectness**: `k` scores per spatial location (k = anchors per location) — "is there an object?"
  - **Box regression**: `4k` coordinates per spatial location — box offsets relative to anchors.
- **Anchors**: Reference boxes of multiple scales/aspect ratios at each location.
- **Proposals**: Top-scoring boxes after NMS become RoIs for the second stage.

## Used By

| Model | Role |
|-------|------|
| Faster R-CNN | The defining component |
| Mask R-CNN | RPN for region proposals |
| Cascade R-CNN | RPN as the first stage |

## Features

- **End-to-end**: RPN + detector trained jointly, no external proposals.
- **Anchor-based**: Multi-scale/aspect-ratio anchors handle diverse object sizes.
- **Shared computation**: Slides over the backbone feature map, reusing features.

## Evolution

- **Predecessor**: Selective Search (external, slow); Edge Boxes.
- **Successor**: Anchor-free detectors (FCOS, CenterNet); DETR (set-based, no RPN).
