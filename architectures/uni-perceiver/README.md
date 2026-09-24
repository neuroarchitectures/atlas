# Uni-Perceiver

## Overview

Uni-Perceiver is a **unified architecture for generic perception**: one shared-parameter Transformer encoder with lightweight modality-specific tokenizers maps vision, language, and vision-language inputs into a **single representation space**. Every task — classification, retrieval, VQA, captioning — is formulated as **maximum-likelihood matching between input and target representations**, which is what enables **zero-shot** transfer to tasks never seen in pretraining, and near-SOTA performance with **1% prompt tuning**.

- **Year:** 2021
- **Authors:** Zhu et al. (BAAI / CUHK)
- **Source:** arXiv:2112.01522 — *Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks*
- **Category:** DL/Unified-Perception

## Key Characteristics

- **Modality-agnostic Transformer encoder** — one set of parameters shared across all modalities and tasks; only the lightweight input tokenizers are modality-specific.
- **Everything in one latent space** — inputs *and* candidate outputs (labels, answers, captions) are encoded into the same space; prediction reduces to similarity-based maximum-likelihood estimation.
- **Masked-prediction pretraining** on large-scale unimodal and multimodal data.
- **Zero-shot**: reasonable performance on unseen tasks with no fine-tuning; **prompt tuning**: ~1% of downstream data reaches near-SOTA; full fine-tuning matches or beats SOTA.
- The direct ancestor of the "one model, many tasks" line (Uni-Perceiver v2, OFA, BEiT-3, Unified-IO, Gato).

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhu_et_al._2021_2112.01522.md`](references/papers/Zhu_et_al._2021_2112.01522.md)

## Related Architectures

See `architecture.md` → Evolution section (predecessors: BERT, CLIP, DETR; successors: Uni-Perceiver v2, OFA, BEiT-3, Unified-IO).
