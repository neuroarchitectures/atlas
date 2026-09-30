# Unifying Structural Proximity and Equivalence for Network Embedding

> Source: `https://arxiv.org/abs/2019.29323`

---

**Authors:** Benyun Shi, Chunpeng Zhou, Hongjun Qiu, Xiaobin Xu, Jiming Liu

**Published:** 2019

**Concepts:** Computer science, Embedding, Node (physics), Vector space, Equivalence (formal languages)

## Abstract

The fundamental purpose of network embedding is to automatically encode each node in a network as a low-dimensional vector, while at the same time preserving certain characteristics of the network. Based on the nodes' embeddings, downstream network analytic tasks such as community mining, node classification, and link prediction, can then be easily implemented using traditional machine learning methods. In recent years, extensive network embedding methods have been proposed based on factorization, random walks, deep learning, and so on. However, most of them focus mainly on preserving the structural proximity of network nodes, where highly interconnected nodes in a network will be represented closely together in the embedded vector space. While in many real-world networks, existing studies have revealed that high-order organizations (e.g., network motifs and graphlets) may be related to specific network functions. In this case, nodes far apart but with a similar organization in a network (i.e., structural equivalence) may have similar network functions. Accordingly, in this paper, we present a hybrid embedding method that unifies both structural proximity and equivalence (SPaE) of a network. Specifically, we adopt the concept of graphlet degree vector (GDV) to measure structural equivalence between network nodes. Through carrying out experiments on both synthetic and real-world datasets, we evaluate the performance of the hybrid embedding method in tasks of node clustering, node classification, and visualization. The results demonstrate that the proposed SPaE method outperforms several state-of-the-art methods when the network analytic tasks are not merely related to structural proximity. Finally, we also conduct experiments to evaluate the flexibility, robustness, and parameter sensitivity of the hybrid embedding method.

**PDF:** https://arxiv.org/pdf/2019.29323