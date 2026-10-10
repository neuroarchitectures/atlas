# Paper (Wen et al. 2019)

> Source: `https://arxiv.org/abs/1910.07099`

---

**Entire Space Multi-Task Modeling via Post-Click Behavior Decomposition for Conversion Rate Prediction**

Hong Wen, Jing Zhang, Yuan Wang, Fuyu Lv, Wentian Bao, Quan Lin, Keping Yang

**Abstract**

Recommender system, as an essential part of modern e-commerce, consists of two fundamental modules, namely Click-Through Rate (CTR) and Conversion Rate (CVR) prediction. While CVR has a direct impact on the purchasing volume, its prediction is well-known challenging due to the Sample Selection Bias (SSB) and Data Sparsity (DS) issues. Although existing methods, typically built on the user sequential behavior path ``impression$\to$click$\to$purchase'', is effective for dealing with SSB issue, they still struggle to address the DS issue due to rare purchase training samples. Observing that users always take several purchase-related actions after clicking, we propose a novel idea of post-click behavior decomposition. Specifically, disjoint purchase-related Deterministic Action (DAction) and Other Action (OAction) are inserted between click and purchase in parallel, forming a novel user sequential behavior graph ``impression$\to$click$\to$D(O)Action$\to$purchase''. Defining model on this graph enables to leverage all the impression samples over the entire space and extra abundant supervised signals from D(O)Action, which will effectively address the SSB and DS issues together. To this end, we devise a novel deep recommendation model named Elaborated Entire Space Supervised Multi-task Model ($ESM^{2}$). According to the conditional probability rule defined on the graph, it employs multi-task learning to predict some decomposed sub-targets in parallel and compose them sequentially to formulate the final CVR. Extensive experiments on both offline and online environments demonstrate the superiority of $ESM^{2}$ over state-of-the-art models. The source code and dataset will be released.

---

- **arXiv ID**: `1910.07099`
- **Published**: 2019-10-15
- **Categories**: cs.LG, cs.IR, stat.ML
- **PDF**: https://arxiv.org/pdf/1910.07099
