# Paper (Fountoulakis et al. 2022)

> Source: `https://arxiv.org/abs/2202.13060`

---

**Graph Attention Retrospective**

Kimon Fountoulakis, Amit Levi, Shenghao Yang, Aseem Baranwal, Aukosh Jagannath

**Abstract**

Graph-based learning is a rapidly growing sub-field of machine learning with applications in social networks, citation networks, and bioinformatics. One of the most popular models is graph attention networks. They were introduced to allow a node to aggregate information from features of neighbor nodes in a non-uniform way, in contrast to simple graph convolution which does not distinguish the neighbors of a node. In this paper, we theoretically study the behaviour of graph attention networks. We prove multiple results on the performance of the graph attention mechanism for the problem of node classification for a contextual stochastic block model. Here, the node features are obtained from a mixture of Gaussians and the edges from a stochastic block model. We show that in an "easy" regime, where the distance between the means of the Gaussians is large enough, graph attention is able to distinguish inter-class from intra-class edges. Thus it maintains the weights of important edges and significantly reduces the weights of unimportant edges. Consequently, we show that this implies perfect node classification. In the "hard" regime, we show that every attention mechanism fails to distinguish intra-class from inter-class edges. In addition, we show that graph attention convolution cannot (almost) perfectly classify the nodes even if intra-class edges could be separated from inter-class edges. Beyond perfect node classification, we provide a positive result on graph attention's robustness against structural noise in the graph. In particular, our robustness result implies that graph attention can be strictly better than both the simple graph convolution and the best linear classifier of node features. We evaluate our theoretical results on synthetic and real-world data.

---

- **arXiv ID**: `2202.13060`
- **Published**: 2022-02-26
- **Categories**: cs.LG, stat.ML
- **PDF**: https://arxiv.org/pdf/2202.13060
