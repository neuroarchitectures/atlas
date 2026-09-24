# EfficientSAM

## Overview

EfficientSAM attacks SAM's cost at the *encoder*, but instead of distilling from a teacher it **pretrains** the small encoder properly: **SAMI** (SAM-leveraged Masked Image Pretraining) trains a lightweight ViT to reconstruct SAM's ViT-H embeddings from a masked image, using SAM's own encoder as the reconstruction target rather than pixels. The pretrained encoder is then paired with SAM's mask decoder and fine-tuned on SA-1B.

- **Year:** 2023
- **Authors:** Xiong et al.
- **Source:** arXiv:2312.00863 — *EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything*
- **Category:** DL/Segmentation

## Key Characteristics

- **SAMI pretraining** — reconstruct SAM image-encoder features (not pixels) from masked inputs, which transfers SAM's segmentation-relevant representation into a tiny ViT.
- **No labels needed** — SAMI runs on ImageNet-1K without annotation; SAM's embeddings are the supervision signal.
- **Lightweight ViT encoders** (Tiny / Small) replace the ViT-H encoder while keeping SAM's prompt encoder and mask decoder.
- **Fine-tuning on SA-1B** aligns the small encoder with the mask decoder for the promptable task.
- **Better fidelity than naive distillation** — the pretraining objective, not the parameter count, is what recovers most of the quality gap.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Xiong_et_al._2023_2312.00863.md`](references/papers/Xiong_et_al._2023_2312.00863.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MAE → SAMI → EfficientSAM; siblings: MobileSAM (decoupled distillation), EdgeSAM (prompt-in-the-loop distillation), FastSAM (CNN); baseline: SAM).
