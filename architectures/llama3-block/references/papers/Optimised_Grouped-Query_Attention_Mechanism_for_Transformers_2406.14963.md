# Paper (Chen et al. 2024)

> Source: `https://arxiv.org/abs/2406.14963`

---

**Optimised Grouped-Query Attention Mechanism for Transformers**

Yuang Chen, Cheng Zhang, Xitong Gao, Robert D. Mullins, George A. Constantinides, Yiren Zhao

**Abstract**

Grouped-query attention (GQA) has been widely adopted in LLMs to mitigate the complexity of multi-head attention (MHA). To transform an MHA to a GQA, neighbour queries in MHA are evenly split into groups where each group shares the value and key layers. In this work, we propose AsymGQA, an activation-informed approach to asymmetrically grouping an MHA to a GQA for better model performance. Our AsymGQA outperforms the GQA within the same model size budget. For example, AsymGQA LLaMA-2-7B has an accuracy increase of 7.5% on MMLU compared to neighbour grouping. Our approach addresses the GQA's trade-off problem between model performance and hardware efficiency.

---

- **arXiv ID**: `2406.14963`
- **Published**: 2024-06-21
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/2406.14963
