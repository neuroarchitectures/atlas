# Paper (Wang et al. 2021)

> Source: `https://arxiv.org/abs/2108.04475`

---

**Localized Graph Collaborative Filtering**

Yiqi Wang, Chaozhuo Li, Mingzheng Li, Wei Jin, Yuming Liu, Hao Sun, Xing Xie, Jiliang Tang

**Abstract**

User-item interactions in recommendations can be naturally de-noted as a user-item bipartite graph. Given the success of graph neural networks (GNNs) in graph representation learning, GNN-based C methods have been proposed to advance recommender systems. These methods often make recommendations based on the learned user and item embeddings. However, we found that they do not perform well wit sparse user-item graphs which are quite common in real-world recommendations. Therefore, in this work, we introduce a novel perspective to build GNN-based CF methods for recommendations which leads to the proposed framework Localized Graph Collaborative Filtering (LGCF). One key advantage of LGCF is that it does not need to learn embeddings for each user and item, which is challenging in sparse scenarios. Alternatively, LGCF aims at encoding useful CF information into a localized graph and making recommendations based on such graph. Extensive experiments on various datasets validate the effectiveness of LGCF especially in sparse scenarios. Furthermore, empirical results demonstrate that LGCF provides complementary information to the embedding-based CF model which can be utilized to boost recommendation performance.

---

- **arXiv ID**: `2108.04475`
- **Published**: 2021-08-10
- **Categories**: cs.IR, cs.AI
- **PDF**: https://arxiv.org/pdf/2108.04475
