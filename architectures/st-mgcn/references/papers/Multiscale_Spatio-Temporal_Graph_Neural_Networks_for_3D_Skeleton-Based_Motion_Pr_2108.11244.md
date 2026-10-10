# Paper (Li et al. 2021)

> Source: `https://arxiv.org/abs/2108.11244`

---

**Multiscale Spatio-Temporal Graph Neural Networks for 3D Skeleton-Based Motion Prediction**

Maosen Li, Siheng Chen, Yangheng Zhao, Ya Zhang, Yanfeng Wang, Qi Tian

**Abstract**

We propose a multiscale spatio-temporal graph neural network (MST-GNN) to predict the future 3D skeleton-based human poses in an action-category-agnostic manner. The core of MST-GNN is a multiscale spatio-temporal graph that explicitly models the relations in motions at various spatial and temporal scales. Different from many previous hierarchical structures, our multiscale spatio-temporal graph is built in a data-adaptive fashion, which captures nonphysical, yet motion-based relations. The key module of MST-GNN is a multiscale spatio-temporal graph computational unit (MST-GCU) based on the trainable graph structure. MST-GCU embeds underlying features at individual scales and then fuses features across scales to obtain a comprehensive representation. The overall architecture of MST-GNN follows an encoder-decoder framework, where the encoder consists of a sequence of MST-GCUs to learn the spatial and temporal features of motions, and the decoder uses a graph-based attention gate recurrent unit (GA-GRU) to generate future poses. Extensive experiments are conducted to show that the proposed MST-GNN outperforms state-of-the-art methods in both short and long-term motion prediction on the datasets of Human 3.6M, CMU Mocap and 3DPW, where MST-GNN outperforms previous works by 5.33% and 3.67% of mean angle errors in average for short-term and long-term prediction on Human 3.6M, and by 11.84% and 4.71% of mean angle errors for short-term and long-term prediction on CMU Mocap, and by 1.13% of mean angle errors on 3DPW in average, respectively. We further investigate the learned multiscale graphs for interpretability.

---

- **arXiv ID**: `2108.11244`
- **Published**: 2021-08-25
- **Categories**: cs.CV, cs.AI
- **PDF**: https://arxiv.org/pdf/2108.11244
