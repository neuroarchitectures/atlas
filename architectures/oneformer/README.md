# OneFormer

## Overview

Earlier panoptic architectures claimed unification but still had to be trained separately per task to reach their best numbers — three training runs, three times the compute, three models. OneFormer targets *true* universality: **one model, trained once**, that beats specialized Mask2Former on all three segmentation tasks. Its ingredients are task-conditioned joint training, a **task token**, and a query-text contrastive loss.

- **Year:** 2022
- **Authors:** Jain et al. (SHI Labs, UIUC, University of Oregon)
- **Source:** arXiv:2211.06220 — *OneFormer: One Transformer to Rule Universal Image Segmentation*
- **Category:** DL/Segmentation

## Key Characteristics

- **Train once, not three times** — a task-conditioned joint training strategy consumes semantic, instance and panoptic ground truth inside a single multi-task process; the authors argue this is the requirement for a genuinely universal framework.
- **Task token** — conditions the model on the task at hand, making it task-dynamic at both training and inference.
- **Query-text contrastive loss** — establishes better inter-task and inter-class distinctions during training.
- **Beats per-task specialists** — one OneFormer model outperforms specialized Mask2Former models on ADE20K, Cityscapes and COCO, despite Mask2Former being trained three times with three times the resources.
- **Backbone-agnostic** — further gains reported with ConvNeXt and DiNAT backbones.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Jain_et_al._2022_2211.06220.md`](references/papers/Jain_et_al._2022_2211.06220.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (panoptic architectures → Mask2Former → OneFormer; siblings: SegFormer, Mask2Former, kMaX-DeepLab; successors: X-Decoder, SEEM, SAM-class promptable models).
