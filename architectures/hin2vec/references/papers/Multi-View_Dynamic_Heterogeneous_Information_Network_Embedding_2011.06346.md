# Paper (Zhang et al. 2020)

> Source: `https://arxiv.org/abs/2011.06346`

---

**Multi-View Dynamic Heterogeneous Information Network Embedding**

Zhenghao Zhang, Jianbin Huang, Qinglin Tan

**Abstract**

Most existing Heterogeneous Information Network (HIN) embedding methods focus on static environments while neglecting the evolving characteristic of realworld networks. Although several dynamic embedding methods have been proposed, they are merely designed for homogeneous networks and cannot be directly applied in heterogeneous environment. To tackle above challenges, we propose a novel framework for incorporating temporal information into HIN embedding, denoted as Multi-View Dynamic HIN Embedding (MDHNE), which can efficiently preserve evolution patterns of implicit relationships from different views in updating node representations over time. We first transform HIN to a series of homogeneous networks corresponding to different views. Then our proposed MDHNE applies Recurrent Neural Network (RNN) to incorporate evolving pattern of complex network structure and semantic relationships between nodes into latent embedding spaces, and thus the node representations from multiple views can be learned and updated when HIN evolves over time. Moreover, we come up with an attention based fusion mechanism, which can automatically infer weights of latent representations corresponding to different views by minimizing the objective function specific for different mining tasks. Extensive experiments clearly demonstrate that our MDHNE model outperforms state-of-the-art baselines on three real-world dynamic datasets for different network mining tasks.

---

- **arXiv ID**: `2011.06346`
- **Published**: 2020-11-12
- **Categories**: cs.SI, cs.AI, cs.LG
- **PDF**: https://arxiv.org/pdf/2011.06346
