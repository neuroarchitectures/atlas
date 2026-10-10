# Paper (Sang et al. 2018)

> Source: `https://arxiv.org/abs/1803.09080`

---

**AAANE: Attention-based Adversarial Autoencoder for Multi-scale Network Embedding**

Lei Sang, Min Xu, Shengsheng Qian, Xindong Wu

**Abstract**

Network embedding represents nodes in a continuous vector space and preserves structure information from the Network. Existing methods usually adopt a "one-size-fits-all" approach when concerning multi-scale structure information, such as first- and second-order proximity of nodes, ignoring the fact that different scales play different roles in the embedding learning. In this paper, we propose an Attention-based Adversarial Autoencoder Network Embedding(AAANE) framework, which promotes the collaboration of different scales and lets them vote for robust representations. The proposed AAANE consists of two components: 1) Attention-based autoencoder effectively capture the highly non-linear network structure, which can de-emphasize irrelevant scales during training. 2) An adversarial regularization guides the autoencoder learn robust representations by matching the posterior distribution of the latent embeddings to given prior distribution. This is the first attempt to introduce attention mechanisms to multi-scale network embedding. Experimental results on real-world networks show that our learned attention parameters are different for every network and the proposed approach outperforms existing state-of-the-art approaches for network embedding.

---

- **arXiv ID**: `1803.09080`
- **Published**: 2018-03-24
- **Categories**: cs.LG, stat.ML
- **PDF**: https://arxiv.org/pdf/1803.09080
