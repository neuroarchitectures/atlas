# Paper (Bick et al. 2026)

> Source: `https://arxiv.org/abs/2602.11374`

---

**Retrieval-Aware Distillation for Transformer-SSM Hybrids**

Aviv Bick, Eric P. Xing, Albert Gu

**Abstract**

State-space models (SSMs) offer efficient sequence modeling but lag behind Transformers on benchmarks that require in-context retrieval. Prior work links this gap to a small set of attention heads, termed Gather-and-Aggregate (G&A), which SSMs struggle to reproduce. We propose *retrieval-aware distillation*, which converts a pretrained Transformer into a hybrid student by preserving only these retrieval-critical heads and distilling the rest into recurrent heads. We identify the essential heads via ablation on a synthetic retrieval task, producing a hybrid with sparse, non-uniform attention placement. We show that preserving **just 2% of attention heads recovers over 95% of teacher performance on retrieval-heavy tasks** (10 heads in a 1B model), requiring far fewer heads than hybrids that retain at least 25%. We further find that large recurrent states often compensate for missing retrieval: once retrieval is handled by these heads, the SSM backbone can be simplified with limited loss, even with an $8\times$ reduction in state dimension. By reducing both the attention cache and the SSM state, the resulting hybrid is $5$--$6\times$ more memory-efficient than comparable hybrids, closing the Transformer--SSM gap at a fraction of the memory cost.

---

- **arXiv ID**: `2602.11374`
- **Published**: 2026-02-11
- **Categories**: cs.LG, cs.AI
- **PDF**: https://arxiv.org/pdf/2602.11374
