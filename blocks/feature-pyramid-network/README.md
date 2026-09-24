# Feature Pyramid Network (FPN)

## Design Philosophy

Build an **in-network feature pyramid** with lateral connections, so the model has a semantically strong feature map at every resolution. The philosophy: a single-scale feature map forces a tradeoff between semantic strength (deep layers) and spatial resolution (shallow layers). FPN gives you both: top-down upsampling merges deep semantics into high-resolution maps.

## Functionality

- **Bottom-up**: The backbone (e.g., ResNet) produces feature maps C2, C3, C4, C5 at decreasing resolution.
- **Top-down**: Nearest-neighbor upsample the deepest feature map by 2×.
- **Lateral connections**: 1×1 conv on the matching-resolution bottom-up map, then element-wise add with the upsampled map.
- **Output**: 3×3 conv on each merged level → pyramid P2, P3, P4, P5 (each semantically strong at its resolution).
- An extra max-pool produces P6 if needed.

## Used By

| Model | Role |
|-------|------|
| Mask R-CNN | Backbone option (ResNet-FPN) |
| Faster R-CNN | Backbone for multi-scale RoI extraction |
| RetinaNet | One-stage detector with FPN |
| Most detection/segmentation models | The standard multi-scale feature backbone |

## Features

- **Multi-scale**: RoIs are assigned to pyramid levels by size, so small and large objects get features at the right scale.
- **Cheap**: Lateral 1×1 convs add minimal cost.
- **Drop-in**: Sits on top of any conv backbone.

## Evolution

- **Predecessor**: Image pyramids (classical); SSD's multi-scale feature maps (but without the top-down merge).
- **Successor**: PANet (adds bottom-up path augmentation); BiFPN (EfficientDet, weighted bidirectional FPN).
