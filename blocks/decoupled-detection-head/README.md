# Decoupled Anchor-Free Detection Head

## Design Philosophy

Classification wants translation-invariant features; box regression wants localization-sensitive ones — sharing one conv path for both limits accuracy. Split the head into separate classification and regression branches, drop anchors, and assign positives by task-aligned quality instead of IoU alone.

## Functionality

- Per FPN level: shared stem → cls branch (per-class logits) and reg branch (ltrb offsets / DFL distributions).
- Task-aligned assignment (TAL): positives chosen by `cls_score^α · iou^β`; in NMS-free variants a one-to-one branch replaces NMS at inference while a one-to-many branch provides dense training signal.

## Used By

| Model | Role |
|-------|------|
| YOLOv11 | Shared head across detect/segment/pose/OBB variants |
| YOLO26 | NMS-free dual head: one-to-many training branch + one-to-one inference branch |

## Features

- **Task specialization** — the single largest accuracy jump of the YOLOv8+ generation.
- **One head, many tasks** — segmentation/pose/OBB reuse the same decoupled design.

## Evolution

- **Predecessor**: YOLOv3/v5 coupled heads, FCOS anchor-free regression.
- **Successor**: YOLO26's dual one-to-one head (constant-cost end-to-end decoding, no NMS).
