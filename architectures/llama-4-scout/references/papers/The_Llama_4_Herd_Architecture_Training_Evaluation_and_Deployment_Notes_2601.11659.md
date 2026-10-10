# Paper (arXiv et al. 2026)

> Source: `https://arxiv.org/abs/2601.11659`

---

**The Llama 4 Herd: Architecture, Training, Evaluation, and Deployment Notes**

Redacted by arXiv

**Abstract**

This document consolidates publicly reported technical details about Metas Llama 4 model family. It summarizes (i) released variants (Scout and Maverick) and the broader herd context including the previewed Behemoth teacher model, (ii) architectural characteristics beyond a high-level MoE description covering routed/shared-expert structure, early-fusion multimodality, and long-context design elements reported for Scout (iRoPE and length generalization strategies), (iii) training disclosures spanning pre-training, mid-training for long-context extension, and post-training methodology (lightweight SFT, online RL, and lightweight DPO) as described in release materials, (iv) developer-reported benchmark results for both base and instruction-tuned checkpoints, and (v) practical deployment constraints observed across major serving environments, including provider-specific context limits and quantization packaging. The manuscript also summarizes licensing obligations relevant to redistribution and derivative naming, and reviews publicly described safeguards and evaluation practices. The goal is to provide a compact technical reference for researchers and practitioners who need precise, source-backed facts about Llama 4.

---

- **arXiv ID**: `2601.11659`
- **Published**: 2026-01-15
- **Categories**: cs.SE, cs.LG
- **PDF**: https://arxiv.org/pdf/2601.11659
