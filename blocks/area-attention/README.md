# Area Attention

## Design Philosophy

Full-map attention is too expensive for detection feature maps; fixed windows lose global context. Partition the map into *areas* — horizontal or vertical strips spanning the whole extent — so each token attends across an entire row/column band: large receptive field at a fraction of the cost, FlashAttention-class kernel friendly.

## Functionality

- Split H×W feature map into `k` horizontal or vertical areas; tokens within an area attend mutually (window = full-width strip).
- Combined with R-ELAN aggregation in the attention-centric backbone stage.

## Used By

| Model | Role |
|-------|------|
| YOLOv12 | Area attention blocks in the backbone/neck attention stages |

## Features

- **Between window and global** — strip extent gives long-range structure (object extents, horizons) at O(k) cost.
- **Orientation choice** adds scale/aspect diversity.

## Evolution

- **Predecessor**: swin-shifted-window-attention (square windows), axial attention.
- **Successor**: YOLO26 dropped attention in favor of pure conv heads — area attention remains the efficiency point of the YOLO attention branch.
