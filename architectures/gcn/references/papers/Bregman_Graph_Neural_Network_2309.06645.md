# Paper (Zhai et al. 2023)

> Source: `https://arxiv.org/abs/2309.06645`

---

**Bregman Graph Neural Network**

Jiayu Zhai, Lequan Lin, Dai Shi, Junbin Gao

**Abstract**

Numerous recent research on graph neural networks (GNNs) has focused on formulating GNN architectures as an optimization problem with the smoothness assumption. However, in node classification tasks, the smoothing effect induced by GNNs tends to assimilate representations and over-homogenize labels of connected nodes, leading to adverse effects such as over-smoothing and misclassification. In this paper, we propose a novel bilevel optimization framework for GNNs inspired by the notion of Bregman distance. We demonstrate that the GNN layer proposed accordingly can effectively mitigate the over-smoothing issue by introducing a mechanism reminiscent of the "skip connection". We validate our theoretical results through comprehensive empirical studies in which Bregman-enhanced GNNs outperform their original counterparts in both homophilic and heterophilic graphs. Furthermore, our experiments also show that Bregman GNNs can produce more robust learning accuracy even when the number of layers is high, suggesting the effectiveness of the proposed method in alleviating the over-smoothing issue.

---

- **arXiv ID**: `2309.06645`
- **Published**: 2023-09-12
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/2309.06645
