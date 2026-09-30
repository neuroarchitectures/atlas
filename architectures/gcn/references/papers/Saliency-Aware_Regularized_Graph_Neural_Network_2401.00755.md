# Paper (Pei et al. 2024)

> Source: `https://arxiv.org/abs/2401.00755`

---

**Saliency-Aware Regularized Graph Neural Network**

Wenjie Pei, Weina Xu, Zongze Wu, Weichao Li, Jinfan Wang, Guangming Lu, Xiangrong Wang

**Abstract**

The crux of graph classification lies in the effective representation learning for the entire graph. Typical graph neural networks focus on modeling the local dependencies when aggregating features of neighboring nodes, and obtain the representation for the entire graph by aggregating node features. Such methods have two potential limitations: 1) the global node saliency w.r.t. graph classification is not explicitly modeled, which is crucial since different nodes may have different semantic relevance to graph classification; 2) the graph representation directly aggregated from node features may have limited effectiveness to reflect graph-level information. In this work, we propose the Saliency-Aware Regularized Graph Neural Network (SAR-GNN) for graph classification, which consists of two core modules: 1) a traditional graph neural network serving as the backbone for learning node features and 2) the Graph Neural Memory designed to distill a compact graph representation from node features of the backbone. We first estimate the global node saliency by measuring the semantic similarity between the compact graph representation and node features. Then the learned saliency distribution is leveraged to regularize the neighborhood aggregation of the backbone, which facilitates the message passing of features for salient nodes and suppresses the less relevant nodes. Thus, our model can learn more effective graph representation. We demonstrate the merits of SAR-GNN by extensive experiments on seven datasets across various types of graph data. Code will be released.

---

- **arXiv ID**: `2401.00755`
- **Published**: 2024-01-01
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/2401.00755
