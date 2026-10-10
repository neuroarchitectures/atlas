# Paper (Liao et al. 2026)

> Source: `https://arxiv.org/abs/2609.21548`

---

**Dual-Interest Sequential Product Recommendation With Multi-Granular SSM**

Shuiying Liao, P. Y. Mok

**Abstract**

Sequential recommendation aims to predict the next item a user will interact with based on their historical behavior. Advances in Transformers have significantly improved sequential recommendation but are still limited by cost efficiency. Although State Space Models (SSMs) have recently enabled efficient long-range modeling, most existing methods encode each item with a single static contextual role, overlooking the phenomenon of item polysemy. In fact, the same item often plays different semantic roles depending on user context, and existing methods are limited in capturing dynamic behavior across different temporal granularities. In this work, we propose DSRec, a novel dual-interest cross-SSM model that explicitly disentangles item roles across long-term and short-term semantic context. Sequential items are encoded into long-term interest embeddings that capture stable preferences via historical aggregation, and a short-term interest branch that emphasizes local session intent modulated by inter-click time intervals. These interest embeddings are processed through distinct SSM encoders: a full-sequence Mamba for long-term modeling, and a time-modulated SSM that dynamically adjusts state evolution based on temporal gaps. To enable effective cross-granularity alignment, we adopt a residual cross-fusion mechanism that exchanges contextual information between the two branches while preserving semantic independence. Experiments on public benchmarks demonstrate that DSRec outperforms other state-of-the-art methods.

---

- **arXiv ID**: `2609.21548`
- **Published**: 2026-09-18
- **Categories**: cs.AI, cs.LG
- **PDF**: https://arxiv.org/pdf/2609.21548
