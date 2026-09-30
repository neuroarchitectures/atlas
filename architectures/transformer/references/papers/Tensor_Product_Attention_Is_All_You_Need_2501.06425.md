# Paper (Zhang et al. 2025)

> Source: `https://arxiv.org/abs/2501.06425`

---

**Tensor Product Attention Is All You Need**

Yifan Zhang, Yifeng Liu, Huizhuo Yuan, Zhen Qin, Yang Yuan, Quanquan Gu, Andrew Chi-Chih Yao

**Abstract**

Scaling language models to handle longer input sequences typically necessitates large key-value (KV) caches, resulting in substantial memory overhead during inference. In this paper, we propose Tensor Product Attention (TPA), a novel attention mechanism that uses tensor decompositions to represent queries, keys, and values compactly, substantially shrinking the KV cache size at inference time. By factorizing these representations into contextual low-rank components and seamlessly integrating with Rotary Position Embedding (RoPE), TPA achieves improved model quality alongside memory efficiency. Based on TPA, we introduce the Tensor ProducT ATTenTion Transformer (T6), a new model architecture for sequence modeling. Through extensive empirical evaluation on language modeling tasks, we demonstrate that T6 surpasses or matches the performance of standard Transformer baselines including Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and Multi-Head Latent Attention (MLA) across various metrics, including perplexity and a range of established evaluation benchmarks. Notably, TPA's memory efficiency and computational efficiency at decoding stage enables processing longer sequences under fixed resource constraints, addressing a critical scalability challenge in modern language models. Project Page: https://github.com/tensorgi/TPA.

---

- **arXiv ID**: `2501.06425`
- **Published**: 2025-01-11
- **Categories**: cs.CL, cs.AI, cs.LG
- **PDF**: https://arxiv.org/pdf/2501.06425
