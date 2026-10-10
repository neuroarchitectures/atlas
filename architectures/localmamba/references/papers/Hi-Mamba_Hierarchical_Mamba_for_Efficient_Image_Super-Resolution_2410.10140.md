# Paper (Qiao et al. 2024)

> Source: `https://arxiv.org/abs/2410.10140`

---

**Hi-Mamba: Hierarchical Mamba for Efficient Image Super-Resolution**

Junbo Qiao, Jincheng Liao, Wei Li, Yulun Zhang, Yong Guo, Yi Wen, Zhangxizi Qiu, Jiao Xie, Jie Hu, Shaohui Lin

**Abstract**

State Space Models (SSM), such as Mamba, have shown strong representation ability in modeling long-range dependency with linear complexity, achieving successful applications from high-level to low-level vision tasks. However, SSM's sequential nature necessitates multiple scans in different directions to compensate for the loss of spatial dependency when unfolding the image into a 1D sequence. This multi-direction scanning strategy significantly increases the computation overhead and is unbearable for high-resolution image processing. To address this problem, we propose a novel Hierarchical Mamba network, namely, Hi-Mamba, for image super-resolution (SR). Hi-Mamba consists of two key designs: (1) The Hierarchical Mamba Block (HMB) assembled by a Local SSM (L-SSM) and a Region SSM (R-SSM) both with the single-direction scanning, aggregates multi-scale representations to enhance the context modeling ability. (2) The Direction Alternation Hierarchical Mamba Group (DA-HMG) allocates the isomeric single-direction scanning into cascading HMBs to enrich the spatial relationship modeling. Extensive experiments demonstrate the superiority of Hi-Mamba across five benchmark datasets for efficient SR. For example, Hi-Mamba achieves a significant PSNR improvement of 0.29 dB on Manga109 for $\times3$ SR, compared to the strong lightweight MambaIR.

---

- **arXiv ID**: `2410.10140`
- **Published**: 2024-10-14
- **Categories**: cs.CV
- **PDF**: https://arxiv.org/pdf/2410.10140
