# Paper (Patil et al. 2020)

> Source: `https://arxiv.org/abs/2012.06547`

---

**LayoutGMN: Neural Graph Matching for Structural Layout Similarity**

Akshay Gadi Patil, Manyi Li, Matthew Fisher, Manolis Savva, Hao Zhang

**Abstract**

We present a deep neural network to predict structural similarity between 2D layouts by leveraging Graph Matching Networks (GMN). Our network, coined LayoutGMN, learns the layout metric via neural graph matching, using an attention-based GMN designed under a triplet network setting. To train our network, we utilize weak labels obtained by pixel-wise Intersection-over-Union (IoUs) to define the triplet loss. Importantly, LayoutGMN is built with a structural bias which can effectively compensate for the lack of structure awareness in IoUs. We demonstrate this on two prominent forms of layouts, viz., floorplans and UI designs, via retrieval experiments on large-scale datasets. In particular, retrieval results by our network better match human judgement of structural layout similarity compared to both IoUs and other baselines including a state-of-the-art method based on graph neural networks and image convolution. In addition, LayoutGMN is the first deep model to offer both metric learning of structural layout similarity and structural matching between layout elements.

---

- **arXiv ID**: `2012.06547`
- **Published**: 2020-12-11
- **Categories**: cs.CV, cs.IR
- **PDF**: https://arxiv.org/pdf/2012.06547
