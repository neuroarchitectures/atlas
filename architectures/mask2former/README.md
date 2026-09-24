# Mask2Former

## Overview

Different segmentation tasks differ only in *semantics* (category vs. instance membership), yet each had its own specialized architecture. Mask2Former is a **universal** architecture — same model, loss and training recipe for panoptic, instance and semantic segmentation — whose key component is **masked attention**: in the transformer decoder, cross-attention is restricted to the predicted mask region rather than the whole feature map.

- **Year:** 2021
- **Authors:** Cheng et al.
- **Source:** arXiv:2112.01527 — *Masked-attention Mask Transformer for Universal Image Segmentation*
- **Category:** DL/Segmentation

## Key Characteristics

- **Masked attention** — cross-attention constrained within each query's predicted mask, so queries extract *localized* features instead of attending globally; this is the design that made convergence fast and accuracy high.
- **One architecture for three tasks** — panoptic, instance and semantic segmentation without changing architecture, loss or training procedure.
- **Mask classification** — predicts a set of binary masks each with a single class, following DETR's set-prediction formulation.
- **Multi-scale features** — a pixel decoder produces a feature pyramid; masked attention is applied per scale in a deformable-detector-like fashion.
- **Beats specialized models** — state of the art on COCO panoptic (57.8 PQ), COCO instance (50.1 AP) and ADE20K semantic (57.7 mIoU) at publication.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Cheng_et_al._2021_2112.01527.md`](references/papers/Cheng_et_al._2021_2112.01527.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (FCN → DETR → MaskFormer → Mask2Former; siblings: SegFormer, OneFormer, kMaX-DeepLab; successors: SAM-class promptable models).
