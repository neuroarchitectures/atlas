# Focal Loss

## Design Philosophy

In dense detection, the extreme foreground/background imbalance (≈10⁴–10⁵ easy negatives per positive) means the loss is dominated by negatives the model already classifies correctly. Focal loss **down-weights well-classified examples** and focuses capacity on hard ones, so a one-stage detector can be trained without the sampling heuristics that two-stage detectors use.

## Functionality

`FL(p_t) = −α_t (1 − p_t)^γ log(p_t)`, where `p_t` is the model's probability for the true class.

- The modulating factor `(1 − p_t)^γ` → 0 for easy examples (`p_t → 1`) and stays near 1 for hard ones.
- `γ` (focusing parameter): 0 reduces to standard CE; γ = 2 is the standard choice; larger γ focuses harder.
- `α_t` balances positive/negative frequency (the α-balanced variant, typically α = 0.25 for the foreground).
- Used with the standard smooth-L1 / IoU box loss; applied per anchor/point.

## Used By

| Model | Role |
|-------|------|
| RetinaNet | Classification loss (the originating use) |
| FCOS, ATSS, GFL | Dense one-stage detectors |
| YOLO-family variants, RTMDet | Objectness / classification variants |
| Long-tail classification, segmentation | Class-imbalance handling |

## Features

- **Automatic hard-example mining** inside the loss — no OHEM sampling needed.
- Two knobs (`α`, `γ`) that are fairly robust across datasets.
- Does not solve localization quality — hence successors that add an IoU-aware term.

## Evolution

- **Predecessors**: cross-entropy; balanced cross-entropy / α-weighting; online hard-example mining (OHEM).
- **Itself**: Focal Loss (Lin et al., 2017, RetinaNet).
- **Successors**: Quality Focal Loss (QFL, continuous IoU labels), Varifocal Loss (VFL, IoU-aware asymmetric weighting, used in RT-DETR-style detectors), Distribution Focal Loss (DFL, in D-FINE and YOLOv8+), and GFL (joint localization quality).
