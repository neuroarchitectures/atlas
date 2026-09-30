# Paper (Gan et al. 2024)

> Source: `https://arxiv.org/abs/2408.00420`

---

**MPT-PAR:Mix-Parameters Transformer for Panoramic Activity Recognition**

Wenqing Gan, Yan Sun, Feiran Liu, Xiangfeng Luo

**Abstract**

The objective of the panoramic activity recognition task is to identify behaviors at various granularities within crowded and complex environments, encompassing individual actions, social group activities, and global activities. Existing methods generally use either parameter-independent modules to capture task-specific features or parameter-sharing modules to obtain common features across all tasks. However, there is often a strong interrelatedness and complementary effect between tasks of different granularities that previous methods have yet to notice. In this paper, we propose a model called MPT-PAR that considers both the unique characteristics of each task and the synergies between different tasks simultaneously, thereby maximizing the utilization of features across multi-granularity activity recognition. Furthermore, we emphasize the significance of temporal and spatial information by introducing a spatio-temporal relation-enhanced module and a scene representation learning module, which integrate the the spatio-temporal context of action and global scene into the feature map of each granularity. Our method achieved an overall F1 score of 47.5\% on the JRDB-PAR dataset, significantly outperforming all the state-of-the-art methods.

---

- **arXiv ID**: `2408.00420`
- **Published**: 2024-08-01
- **Categories**: cs.CV, cs.AI
- **PDF**: https://arxiv.org/pdf/2408.00420
