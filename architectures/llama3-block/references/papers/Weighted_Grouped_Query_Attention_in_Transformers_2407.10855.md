# Paper (Chinnakonduru et al. 2024)

> Source: `https://arxiv.org/abs/2407.10855`

---

**Weighted Grouped Query Attention in Transformers**

Sai Sena Chinnakonduru, Astarag Mohapatra

**Abstract**

The attention mechanism forms the foundational blocks for transformer language models. Recent approaches show that scaling the model achieves human-level performance. However, with increasing demands for scaling and constraints on hardware memory, the inference costs of these models remain high. To reduce the inference time, Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) were proposed in (Shazeer, 2019) and (Ainslieet al., 2023) respectively. In this paper, we propose a variation of Grouped-Query Attention, termed Weighted Grouped-Query Attention (WGQA). We introduced new learnable parameters for each key and value head in the T5 decoder attention blocks, enabling the model to take a weighted average during finetuning. Our model achieves an average of 0.53% improvement over GQA, and the performance converges to traditional Multi-head attention (MHA) with no additional overhead during inference. We evaluated the introduction of these parameters and subsequent finetuning informs the model about the grouping mechanism during training, thereby enhancing performance. Additionally, we demonstrate the scaling laws in our analysis by comparing the results between T5-small and T5-base architecture.

---

- **arXiv ID**: `2407.10855`
- **Published**: 2024-07-15
- **Categories**: cs.CL, cs.AI
- **PDF**: https://arxiv.org/pdf/2407.10855
