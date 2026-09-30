# Paper (Schneider et al. 2025)

> Source: `https://arxiv.org/abs/2507.06204`

---

**Differential Mamba**

Nadav Schneider, Itamar Zimerman, Eliya Nachmani

**Abstract**

Sequence models like Transformers and RNNs often overallocate attention to irrelevant context, leading to noisy intermediate representations. This degrades LLM capabilities by promoting hallucinations, weakening long-range and retrieval abilities, and reducing robustness. Recent work has shown that differential design can mitigate this issue in Transformers, improving their effectiveness across various applications. In this paper, we explore whether these techniques, originally developed for Transformers, can be applied to Mamba, a recent architecture based on selective state-space layers that achieves Transformer-level performance with greater efficiency. We show that a naive adaptation of differential design to Mamba is insufficient and requires careful architectural modifications. To address this, we introduce a novel differential mechanism for Mamba, empirically validated on language modeling benchmarks, demonstrating improved retrieval capabilities and superior performance over vanilla Mamba. Finally, we conduct extensive ablation studies and empirical analyses to justify our design choices and provide evidence that our approach effectively mitigates the overallocation problem in Mamba-based models. Our code is publicly available: https://github.com/NadavSc/Diff-Mamba

---

- **arXiv ID**: `2507.06204`
- **Published**: 2025-07-08
- **Categories**: cs.LG, cs.AI, cs.CL
- **PDF**: https://arxiv.org/pdf/2507.06204
