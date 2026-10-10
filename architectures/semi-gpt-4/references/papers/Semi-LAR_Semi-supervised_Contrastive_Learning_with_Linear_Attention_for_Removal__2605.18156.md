# Paper (Zhu et al. 2026)

> Source: `https://arxiv.org/abs/2605.18156`

---

**Semi-LAR: Semi-supervised Contrastive Learning with Linear Attention for Removal of Nighttime Flares**

Xiyu Zhu, Wei Wang, Kui Jiang, Zhengguo Li

**Abstract**

Lens flare removal is challenging due to the large spatial extent of flare artifacts and their entanglement with scene structures, while existing methods heavily rely on large-scale paired data. We propose a semi-supervised flare removal framework that enables stable learning from unlabeled images by jointly addressing pseudo-label reliability and representation discrimination. We propose an adaptive pseudo-label repository that progressively refines pseudo supervision through no-reference quality assessment, momentum-based updates, and invalid label filtering, effectively mitigating error accumulation. Moreover, we propose a flare-aware contrastive loss that explicitly treats flare-contaminated inputs as negatives and performs patch-level contrastive learning, encouraging representations that are discriminative against flare patterns while remaining consistent with reliable pseudo targets. Extensive experiments on multiple flare benchmarks demonstrate that the proposed framework is model-agnostic and consistently improves performance and robustness.

---

- **arXiv ID**: `2605.18156`
- **Published**: 2026-05-18
- **Categories**: cs.CV
- **PDF**: https://arxiv.org/pdf/2605.18156
