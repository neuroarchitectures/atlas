# Paper (Ji et al. 2023)

> Source: `https://arxiv.org/abs/2308.04067`

---

**Online Distillation-enhanced Multi-modal Transformer for Sequential Recommendation**

Wei Ji, Xiangyan Liu, An Zhang, Yinwei Wei, Yongxin Ni, Xiang Wang

**Abstract**

Multi-modal recommendation systems, which integrate diverse types of information, have gained widespread attention in recent years. However, compared to traditional collaborative filtering-based multi-modal recommendation systems, research on multi-modal sequential recommendation is still in its nascent stages. Unlike traditional sequential recommendation models that solely rely on item identifier (ID) information and focus on network structure design, multi-modal recommendation models need to emphasize item representation learning and the fusion of heterogeneous data sources. This paper investigates the impact of item representation learning on downstream recommendation tasks and examines the disparities in information fusion at different stages. Empirical experiments are conducted to demonstrate the need to design a framework suitable for collaborative learning and fusion of diverse information. Based on this, we propose a new model-agnostic framework for multi-modal sequential recommendation tasks, called Online Distillation-enhanced Multi-modal Transformer (ODMT), to enhance feature interaction and mutual learning among multi-source input (ID, text, and image), while avoiding conflicts among different features during training, thereby improving recommendation accuracy. To be specific, we first introduce an ID-aware Multi-modal Transformer module in the item representation learning stage to facilitate information interaction among different features. Secondly, we employ an online distillation training strategy in the prediction optimization stage to make multi-source data learn from each other and improve prediction robustness. Experimental results on a stream media recommendation dataset and three e-commerce recommendation datasets demonstrate the effectiveness of the proposed two modules, which is approximately 10% improvement in performance compared to baseline models.

---

- **arXiv ID**: `2308.04067`
- **Published**: 2023-08-08
- **Categories**: cs.IR
- **PDF**: https://arxiv.org/pdf/2308.04067
