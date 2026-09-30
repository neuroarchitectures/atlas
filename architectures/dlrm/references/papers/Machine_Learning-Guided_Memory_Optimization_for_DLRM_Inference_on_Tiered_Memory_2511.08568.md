# Paper (Ren et al. 2025)

> Source: `https://arxiv.org/abs/2511.08568`

---

**Machine Learning-Guided Memory Optimization for DLRM Inference on Tiered Memory**

Jie Ren, Bin Ma, Shuangyan Yang, Benjamin Francis, Ehsan K. Ardestani, Min Si, Dong Li

**Abstract**

Deep learning recommendation models (DLRMs) are widely used in industry, and their memory capacity requirements reach the terabyte scale. Tiered memory architectures provide a cost-effective solution but introduce challenges in embedding-vector placement due to complex embedding-access patterns. We propose RecMG, a machine learning (ML)-guided system for vector caching and prefetching on tiered memory. RecMG accurately predicts accesses to embedding vectors with long reuse distances or few reuses. The design of RecMG focuses on making ML feasible in the context of DLRM inference by addressing unique challenges in data labeling and navigating the search space for embedding-vector placement. By employing separate ML models for caching and prefetching, plus a novel differentiable loss function, RecMG narrows the prefetching search space and minimizes on-demand fetches. Compared to state-of-the-art temporal, spatial, and ML-based prefetchers, RecMG reduces on-demand fetches by 2.2x, 2.8x, and 1.5x, respectively. In industrial-scale DLRM inference scenarios, RecMG effectively reduces end-to-end DLRM inference time by up to 43%.

---

- **arXiv ID**: `2511.08568`
- **Published**: 2025-11-11
- **Categories**: cs.PF
- **PDF**: https://arxiv.org/pdf/2511.08568
