# Ultralytics YOLO26

## Overview

YOLO26 is Ultralytics' unification of the real-time stack: **NMS-free end-to-end detection** via a dual head (one-to-many for training, one-to-one for inference), **Distribution Focal Loss removed**, and a training recipe built on **ProgLoss** (progressive loss balancing), **STAL** (small-target-aware label assignment) and the **MuSGD** optimizer. One model family covers detection, segmentation, pose and OBB.

- **Year:** 2026
- **Authors:** Jocher et al. (Ultralytics)
- **Source:** arXiv:2606.03748 — *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*
- **Category:** DL/Detection

## Key Characteristics

- **NMS-free end-to-end decoding** — a one-to-one branch produces the final predictions directly; latency no longer depends on how many objects are in the scene.
- **Dual head** — a one-to-many branch supplies dense supervision during training, the one-to-one branch is used at inference; a fallback to the non-E2E path exists for incompatible deployments.
- **DFL removed** — dropping Distribution Focal Loss simplifies export and quantization, which is the stated motivation.
- **ProgLoss + STAL** — progressive loss weighting and small-target-aware assignment, targeting the small-object weakness of earlier versions.
- **MuSGD** — a momentum-unified SGD variant used as the training optimizer.
- **Unified multi-task** — detection, segmentation, pose and OBB from the same backbone and neck.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Jocher_et_al._2026_2606.03748.md`](references/papers/Jocher_et_al._2026_2606.03748.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (YOLOv11 → YOLOv12 → YOLO26; siblings: RT-DETRv3, D-FINE, DEYO (also NMS-free DETR), LW-DETR).
