# Paper (Li et al. 2020)

> Source: `https://arxiv.org/abs/2010.04408`

---

**Dirichlet Graph Variational Autoencoder**

Jia Li, Tomasyu Yu, Jiajin Li, Honglei Zhang, Kangfei Zhao, YU Rong, Hong Cheng, Junzhou Huang

**Abstract**

Graph Neural Networks (GNNs) and Variational Autoencoders (VAEs) have been widely used in modeling and generating graphs with latent factors. However, there is no clear explanation of what these latent factors are and why they perform well. In this work, we present Dirichlet Graph Variational Autoencoder (DGVAE) with graph cluster memberships as latent factors. Our study connects VAEs based graph generation and balanced graph cut, and provides a new way to understand and improve the internal mechanism of VAEs based graph generation. Specifically, we first interpret the reconstruction term of DGVAE as balanced graph cut in a principled way. Furthermore, motivated by the low pass characteristics in balanced graph cut, we propose a new variant of GNN named Heatts to encode the input graph into cluster memberships. Heatts utilizes the Taylor series for fast computation of heat kernels and has better low pass characteristics than Graph Convolutional Networks (GCN). Through experiments on graph generation and graph clustering, we demonstrate the effectiveness of our proposed framework.

---

- **arXiv ID**: `2010.04408`
- **Published**: 2020-10-09
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/2010.04408
