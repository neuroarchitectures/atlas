# Paper (Chen et al. 2018)

> Source: `https://arxiv.org/abs/1801.10247`

---

**FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling**

Jie Chen, Tengfei Ma, Cao Xiao

**Abstract**

The graph convolutional networks (GCN) recently proposed by Kipf and Welling are an effective graph model for semi-supervised learning. This model, however, was originally designed to be learned with the presence of both training and test data. Moreover, the recursive neighborhood expansion across layers poses time and memory challenges for training with large, dense graphs. To relax the requirement of simultaneous availability of test data, we interpret graph convolutions as integral transforms of embedding functions under probability measures. Such an interpretation allows for the use of Monte Carlo approaches to consistently estimate the integrals, which in turn leads to a batched training scheme as we propose in this work---FastGCN. Enhanced with importance sampling, FastGCN not only is efficient for training but also generalizes well for inference. We show a comprehensive set of experiments to demonstrate its effectiveness compared with GCN and related models. In particular, training is orders of magnitude more efficient while predictions remain comparably accurate.

---

- **arXiv ID**: `1801.10247`
- **Published**: 2018-01-30
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/1801.10247
