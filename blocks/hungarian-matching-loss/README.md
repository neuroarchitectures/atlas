# Hungarian (Bipartite) Matching Loss

## Design Philosophy

A detector should emit a **set**, not a ranked list of candidates — so duplicates produced by overlapping anchors must be suppressed by the *training objective* rather than by NMS. The Hungarian loss first finds a **one-to-one assignment** between `N` predictions and ground-truth objects with minimum total cost, then supervises each matched pair. Unmatched predictions are pushed to "no object".

## Functionality

1. **Matching**: solve `argmin_σ Σ_i L_match(y_i, ŷ_{σ(i)})` over permutations `σ` with the Hungarian algorithm, where `L_match` combines classification cost (and, in later variants, box and IoU cost).
2. **Loss** on matched pairs: classification (CE or focal) + box regression (L1) + GIoU.
3. **Cardinality handling**: the prediction set is padded to a fixed size `N`; unmatched slots are supervised as ∅.
4. DETR's original recipe: **3× weight on L1**, auxiliary losses on every decoder layer, and separate treatment of the ∅ class.

## Used By

| Model | Role |
|-------|------|
| DETR | The originating set-prediction objective |
| Deformable DETR, DAB-DETR, DN-DETR, DINO | Same, with query/denoising variants |
| RT-DETR | With IoU-aware classification and uncertainty-minimal query selection |
| Grounding DINO | With contrastive region-text classification cost |
| MOTR / TrackFormer, MaskFormer / Mask2Former | Extended to tracking and mask classification |

## Features

- **NMS-free, duplicate-free** outputs — post-processing becomes trivial.
- **Permutation-invariant**: only the assignment matters, not prediction order.
- Cost: matching is per-image per-step but tiny (scipy/`linear_sum_assignment`); the real cost is that it makes training harder, needing **long schedules** and tricks (query denoising, deformable attention) to converge.

## Evolution

- **Predecessors**: anchor/assignment-based losses (IoU-threshold matching, ATSS, SimOTA) plus NMS post-processing.
- **Itself**: Hungarian matching loss (Carion et al., DETR, 2020).
- **Successors**: Hungarian matching with **query denoising** (DN-DETR, DINO) and contrastive denoising to stabilize and accelerate training; **Hungarian-free** one-to-many assignment variants (e.g. DEYO, Group DETR) that assign multiple queries per object.
