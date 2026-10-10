# Paper (Xu et al. 2020)

> Source: `https://arxiv.org/abs/2005.08008`

---

**Graph Partitioning and Graph Neural Network based Hierarchical Graph Matching for Graph Similarity Computation**

Haoyan Xu, Ziheng Duan, Jie Feng, Runjian Chen, Qianru Zhang, Zhongbin Xu, Yueyang Wang

**Abstract**

Graph similarity computation aims to predict a similarity score between one pair of graphs to facilitate downstream applications, such as finding the most similar chemical compounds similar to a query compound or Fewshot 3D Action Recognition. Recently, some graph similarity computation models based on neural networks have been proposed, which are either based on graph-level interaction or node-level comparison. However, when the number of nodes in the graph increases, it will inevitably bring about reduced representation ability or high computation cost. Motivated by this observation, we propose a graph partitioning and graph neural network-based model, called PSimGNN, to effectively resolve this issue. Specifically, each of the input graphs is partitioned into a set of subgraphs to extract the local structural features directly. Next, a novel graph neural network with an attention mechanism is designed to map each subgraph into an embedding vector. Some of these subgraph pairs are automatically selected for node-level comparison to supplement the subgraph-level embedding with fine-grained information. Finally, coarse-grained interaction information among subgraphs and fine-grained comparison information among nodes in different subgraphs are integrated to predict the final similarity score. Experimental results on graph datasets with different graph sizes demonstrate that PSimGNN outperforms state-of-the-art methods in graph similarity computation tasks using approximate Graph Edit Distance (GED) as the graph similarity metric.

---

- **arXiv ID**: `2005.08008`
- **Published**: 2020-05-16
- **Categories**: cs.LG, cs.SI, stat.ML
- **PDF**: https://arxiv.org/pdf/2005.08008
