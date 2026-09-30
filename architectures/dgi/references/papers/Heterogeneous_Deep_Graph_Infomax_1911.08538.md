# Paper (Ren et al. 2019)

> Source: `https://arxiv.org/abs/1911.08538`

---

**Heterogeneous Deep Graph Infomax**

Yuxiang Ren, Bo Liu, Chao Huang, Peng Dai, Liefeng Bo, Jiawei Zhang

**Abstract**

Graph representation learning is to learn universal node representations that preserve both node attributes and structural information. The derived node representations can be used to serve various downstream tasks, such as node classification and node clustering. When a graph is heterogeneous, the problem becomes more challenging than the homogeneous graph node learning problem. Inspired by the emerging information theoretic-based learning algorithm, in this paper we propose an unsupervised graph neural network Heterogeneous Deep Graph Infomax (HDGI) for heterogeneous graph representation learning. We use the meta-path structure to analyze the connections involving semantics in heterogeneous graphs and utilize graph convolution module and semantic-level attention mechanism to capture local representations. By maximizing local-global mutual information, HDGI effectively learns high-level node representations that can be utilized in downstream graph-related tasks. Experiment results show that HDGI remarkably outperforms state-of-the-art unsupervised graph representation learning methods on both classification and clustering tasks. By feeding the learned representations into a parametric model, such as logistic regression, we even achieve comparable performance in node classification tasks when comparing with state-of-the-art supervised end-to-end GNN models.

---

- **arXiv ID**: `1911.08538`
- **Published**: 2019-11-19
- **Categories**: cs.LG, cs.SI, stat.ML
- **PDF**: https://arxiv.org/pdf/1911.08538
