# GIoU / CIoU / DIoU Loss

## Design Philosophy

L1/L2 box regression optimizes coordinates that are **not** correlated with detection quality, and it is undefined for non-overlapping boxes (zero IoU gives zero gradient). IoU-based losses optimize the actual metric, but raw IoU has a flat gradient when boxes do not overlap. GIoU fixes the non-overlap case; DIoU and CIoU add center distance and aspect-ratio terms for faster, more accurate convergence.

## Functionality

- **IoU**: `1 − IoU`. Scale-invariant, but zero gradient for disjoint boxes.
- **GIoU**: `1 − IoU + (|C \ (B ∪ B_gt)| / |C|)`, where `C` is the smallest enclosing box — gives a gradient even when boxes are disjoint.
- **DIoU**: `1 − IoU + ρ²(b, b_gt) / c²` — penalizes center distance `ρ` normalized by the enclosing diagonal `c`. Converges faster than GIoU.
- **CIoU**: DIoU plus an aspect-ratio consistency term `α·v`. The usual default in YOLO-family detectors.
- Typically **combined with L1** (DETR uses L1 + GIoU) and used together with an IoU-aware classification target.

## Used By

| Model | Role |
|-------|------|
| DETR / Deformable DETR | Box regression loss (L1 + GIoU) |
| YOLOv5–v11, RTMDet | CIoU as the box loss |
| RT-DETR / DINO / D-FINE | IoU-aware variants (with uncertainty/DFL terms) |
| Instance segmentation / rotated detection | Extended to mask-IoU and rotated-IoU forms |

## Features

- **Metric-aligned**: optimizes what is evaluated (IoU at a threshold).
- **Scale-invariant** — no bias toward large boxes (unlike L2).
- **Non-overlap gradient** (GIoU+) — solves the classic L2 "cannot push a disjoint box" problem.
- Not differentiable everywhere in its rotated/polygon forms; CIoU's aspect term can be unstable at extreme ratios.

## Evolution

- **Predecessors**: L2 (smooth-L1) coordinate regression; raw IoU loss.
- **Itself**: GIoU (Rezatofighi et al., 2019), then DIoU and CIoU (Zheng et al., 2020).
- **Successors**: EIoU / SIoU / Wise-IoU (focal-style gradient reweighting by example quality), and **distribution-based** regression (DFL / D-FINE) which models box edges as discrete distributions instead of using a single IoU term.
