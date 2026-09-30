# Paper (Zhang et al. 2026)

> Source: `https://arxiv.org/abs/2602.09386`

---

**SMES: Towards Scalable Multi-Task Recommendation via Expert Sparsity**

Yukun Zhang, Si Dong, Xu Wang, Bo Chen, Qinglin Jia, Shengzhe Wang, Jinlong Jiao, Runhan Li, Jiaqing Liu, Chaoyi Ma et al.

**Abstract**

Industrial recommender systems typically rely on multi-task learning to estimate diverse user feedback signals and aggregate them for ranking. Recent advances in model scaling have shown promising gains in recommendation. However, naively increasing model capacity imposes prohibitive online inference costs and often yields diminishing returns for sparse tasks with skewed label distributions. This mismatch between uniform parameter scaling and heterogeneous task capacity demands poses a fundamental challenge for scalable multi-task recommendation. In this work, we investigate parameter sparsification as a principled scaling paradigm and identify two critical obstacles when applying sparse Mixture-of-Experts (MoE) to multi-task recommendation: exploded expert activation that undermines instance-level sparsity and expert load skew caused by independent task-wise routing. To address these challenges, we propose SMES, a scalable sparse MoE framework with progressive expert routing. SMES decomposes expert activation into a task-shared expert subset jointly selected across tasks and task-adaptive private experts, explicitly bounding per-instance expert execution while preserving task-specific capacity. In addition, SMES introduces a global multi-gate load-balancing regularizer that stabilizes training by regulating aggregated expert utilization across all tasks. SMES has been deployed in Kuaishou large-scale short-video services, supporting over 400 million daily active users. Extensive online experiments demonstrate stable improvements, with GAUC gain of 0.29% and a 0.31% uplift in user watch time.

---

- **arXiv ID**: `2602.09386`
- **Published**: 2026-02-10
- **Categories**: cs.IR
- **PDF**: https://arxiv.org/pdf/2602.09386
