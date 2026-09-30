# Paper (Wu et al. 2026)

> Source: `https://arxiv.org/abs/2605.11934`

---

**Interactive State Space Model with Cross-Modal Local Scanning for Depth Super-Resolution**

Chen Wu, Ling Wang, Zhuoran Zheng, Xiangyu Chen, Jingyuan Xia, Weidong Jiang, Jiantao Zhou

**Abstract**

Guided depth super-resolution (GDSR) reconstructs HR depth maps from LR inputs with HR RGB guidance. Existing methods either model each modality independently or rely on computationally expensive attention mechanisms with quadratic complexity, hindering the establishment of efficient and semantically interactive joint representations. In this paper, we observe that feature maps from different modalities exhibit semantic-level correlations during feature extraction. This motivates us to develop a more flexible approach enabling dense, semantically-aware deep interactions between modalities. To this end, we propose a novel GDSR framework centered around the Interactive State Space Model. Specifically, we design a cross-modal local scanning mechanism that enables fine-grained semantic interactions between RGB and depth features. Leveraging the Mamba architecture, our framework achieves global modeling with linear complexity. Furthermore, a cross-modal matching transform module is introduced to enhance interactive modeling quality by utilizing representative features from both modalities. Extensive experiments demonstrate competitive performance against state-of-the-art methods.

---

- **arXiv ID**: `2605.11934`
- **Published**: 2026-05-12
- **Categories**: cs.CV
- **PDF**: https://arxiv.org/pdf/2605.11934
