# Paper (Huang et al. 2019)

> Source: `https://arxiv.org/abs/1906.10958`

---

**Signed Graph Attention Networks**

Junjie Huang, Huawei Shen, Liang Hou, Xueqi Cheng

**Abstract**

Graph or network data is ubiquitous in the real world, including social networks, information networks, traffic networks, biological networks and various technical networks. The non-Euclidean nature of graph data poses the challenge for modeling and analyzing graph data. Recently, Graph Neural Network (GNNs) are proposed as a general and powerful framework to handle tasks on graph data, e.g., node embedding, link prediction and node classification. As a representative implementation of GNNs, Graph Attention Networks (GATs) are successfully applied in a variety of tasks on real datasets. However, GAT is designed to networks with only positive links and fails to handle signed networks which contain both positive and negative links. In this paper, we propose Signed Graph Attention Networks (SiGATs), generalizing GAT to signed networks. SiGAT incorporates graph motifs into GAT to capture two well-known theories in signed network research, i.e., balance theory and status theory. In SiGAT, motifs offer us the flexible structural pattern to aggregate and propagate messages on the signed network to generate node embeddings. We evaluate the proposed SiGAT method by applying it to the signed link prediction task. Experimental results on three real datasets demonstrate that SiGAT outperforms feature-based method, network embedding method and state-of-the-art GNN-based methods like signed graph convolutional network (SGCN).

---

- **arXiv ID**: `1906.10958`
- **Published**: 2019-06-26
- **Categories**: cs.SI, cs.LG, physics.soc-ph
- **PDF**: https://arxiv.org/pdf/1906.10958
